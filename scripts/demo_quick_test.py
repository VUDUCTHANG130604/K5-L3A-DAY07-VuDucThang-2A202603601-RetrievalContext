"""Quick test demo for 3 retrieval methods without PowerShell quoting issues."""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.lexical import BM25Retriever
from src.retrieval.transformation import QueryTransformer
from src.retrieval.dense import DenseRetriever

def main():
    print("==================================================")
    print("QUICK RETRIEVAL DEMO (3 METHODS)")
    print("==================================================\n")

    chunks = json.load(open("data/chunks.json", encoding="utf-8"))
    query = "Viêm phế quản cấp được định nghĩa như thế nào trong hướng dẫn của Bộ Y tế?"
    print(f"Original Query: '{query}'\n")

    # 1. Lexical BM25
    print("1. [BM25 Lexical Retrieval]")
    bm25 = BM25Retriever(chunks)
    bm25_res = bm25.search(query, top_k=3)
    for r in bm25_res:
        txt_preview = r["text"][:90].replace("\n", " ")
        print(f"   - [{r['id']}] Score: {r['score']:.3f} | Page {r['page']}: {txt_preview}...")
    print()

    # 2. Query Transformation
    print("2. [Query Transformation]")
    qt = QueryTransformer()
    transformed_query = qt.transform(query, method="expand")
    print(f"   - Transformed: '{transformed_query}'\n")

    # 3. Dense Retrieval
    print("3. [Dense Retrieval - EmbeddingGemma-2]")
    dense = DenseRetriever(chunks=chunks)
    dense_res = dense.search(query, top_k=3)
    for r in dense_res:
        txt_preview = r["text"][:90].replace("\n", " ")
        print(f"   - [{r['id']}] Cosine Sim: {r['score']:.3f} | Page {r['page']}: {txt_preview}...")
    print()

    # 4. Dense with Transformed Query
    print("4. [Query Transformation + Dense Retrieval]")
    qt_dense_res = dense.search(transformed_query, top_k=3)
    for r in qt_dense_res:
        txt_preview = r["text"][:90].replace("\n", " ")
        print(f"   - [{r['id']}] Cosine Sim: {r['score']:.3f} | Page {r['page']}: {txt_preview}...")

if __name__ == "__main__":
    main()
