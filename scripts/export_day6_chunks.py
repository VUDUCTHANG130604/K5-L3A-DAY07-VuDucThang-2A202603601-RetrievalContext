"""Export 126 Strategy B chunks from Day 6 directly into Day 7 data format."""
import json
import sys
from pathlib import Path

# Paths
current_dir = Path(__file__).resolve().parent.parent
day6_starter = Path(r"d:\AI Thực chiến\Lad 6\K5-L3A-Day06-VuDucThang-03601-DataIngestionEngineering\starter")

sys.path.insert(0, str(day6_starter))
from src.ingestion import load_pdf
from src.chunking import by_paragraph

def export_chunks():
    pdf_path = day6_starter / "data" / "Huong-dan-chan-doan-va-dieu-tri-benh-ho-hap-theo-Bo-Y-te-vnras.pdf"
    document_id = "byt-benh-ho-hap-2012"
    source = "Huong-dan-chan-doan-va-dieu-tri-benh-ho-hap-theo-Bo-Y-te-vnras.pdf"

    print(f"Loading PDF from: {pdf_path}")
    pages = load_pdf(pdf_path, document_id=document_id, source=source)
    
    # Range 30 to 75 as defined in Day 6
    clinical_pages = [p for p in pages if 30 <= p.page <= 75]
    print(f"Loaded {len(clinical_pages)} clinical pages (30-75).")

    chunks_data = []
    for page in clinical_pages:
        page_chunks = by_paragraph(page, max_size=800)
        for c in page_chunks:
            meta = dict(c.metadata)
            meta["disease"] = meta.get("disease", "")
            meta["strategy"] = "B"
            chunks_data.append({
                "id": c.chunk_id,
                "text": c.text,
                "source_id": c.document_id,
                "source": c.source,
                "page": c.page,
                "metadata": meta
            })

    output_path = current_dir / "data" / "chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported {len(chunks_data)} chunks to {output_path}")

if __name__ == "__main__":
    export_chunks()
