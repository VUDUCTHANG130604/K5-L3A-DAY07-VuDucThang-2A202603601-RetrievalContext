"""Query Transformation Module for Day 7 Medical Retrieval."""
import re
import unicodedata

# Vietnamese medical terminology domain expansion mapping
MEDICAL_SYNONYMS = {
    "định nghĩa": ["khái niệm", "đại cương", "tình trạng viêm"],
    "nguyên tắc điều trị": ["hướng xử trí", "phác đồ điều trị", "tự khỏi"],
    "phòng ngừa": ["dự phòng", "tiêm phòng", "vaccin", "phế cầu", "cúm"],
    "viêm phổi cộng đồng": ["viêm phổi mắc phải ở cộng đồng", "nhiễm khuẩn nhu mô phổi", "ngoài bệnh viện"],
    "viêm phổi bệnh viện": ["viêm phổi mắc phải ở bệnh viện", "sau khi nhập viện", "thở máy"],
    "áp xe phổi": ["ổ mủ", "nhu mô phổi", "hoại tử", "staphylococcus", "vi khuẩn kỵ khí"],
    "copd": ["bệnh phổi tắc nghẽn mạn tính", "tắc nghẽn đường thở", "fev1", "fvc"],
    "đợt cấp copd": ["đợt bùng phát", "khó thở tăng", "thay đổi màu sắc đờm", "anthonisen"],
    "thở oxy": ["lưu lượng", "gọng kính", "spo2", "khí máu động mạch", "paio2"],
    "kháng sinh": ["cefotaxim", "ceftriaxon", "amoxicillin", "clavulanic", "ampicillin", "sulbactam"],
    "corticoid": ["toàn thân", "prednisolon", "methylprednisolon", "chỉ định"],
}

class QueryTransformer:
    """Transforms raw user queries to enhance retrieval performance."""

    def __init__(self):
        pass

    def clean_query(self, query: str) -> str:
        """Remove conversation filler words while keeping essential medical keywords."""
        text = unicodedata.normalize("NFKC", query).strip()
        # Remove typical conversational fillers in Vietnamese
        fillers = [
            r"được định nghĩa như thế nào trong hướng dẫn của bộ y tế\??",
            r"được định nghĩa như thế nào\??",
            r"trong hướng dẫn của bộ y tế\??",
            r"theo hướng dẫn của bộ y tế\??",
            r"theo bộ y tế\??",
            r"như thế nào\??",
            r"là gì\??",
            r"bao nhiêu\??",
            r"được thực hiện ra sao\??",
            r"ra sao\??",
            r"xin cho biết",
            r"hãy cho biết",
        ]
        for f in fillers:
            text = re.sub(f, "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def expand_medical_terms(self, query: str) -> str:
        """Add domain-specific medical synonyms and concepts to the query."""
        clean = self.clean_query(query)
        q_lower = clean.lower()

        expanded_terms = []
        for key, synonyms in MEDICAL_SYNONYMS.items():
            if key in q_lower:
                expanded_terms.extend(synonyms)

        if expanded_terms:
            unique_terms = list(dict.fromkeys(expanded_terms))
            return f"{clean} {' '.join(unique_terms[:6])}"
        return clean

    def transform(self, query: str, method: str = "expand") -> str:
        """Execute the query transformation strategy.
        
        Methods:
        - 'clean': Removes stop words and polite fillers.
        - 'expand': Expands domain-specific medical terms.
        """
        if method == "clean":
            return self.clean_query(query)
        elif method == "expand":
            return self.expand_medical_terms(query)
        return query
