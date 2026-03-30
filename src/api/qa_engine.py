"""
=============================================================================
QA Engine — Grounded LLM Question Answering over Extracted Receipt Data
=============================================================================
Design decisions:
  - GROUNDED QA: LLM receives only the extracted JSON as context.
    It cannot hallucinate information not in the receipt.
  - RETRIEVAL: For multi-receipt queries, FTS5 search finds relevant receipts
    before passing to LLM. Limits context window usage.
  - PROVIDER: Uses Google Gemini Flash (free tier) by default.
    Falls back to a simple rule-based matcher if no API key is set.
=============================================================================
"""

import json
import os
import re
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class QAEngine:

    def __init__(self, store):
        self.store = store
        self._setup_llm()

    def _setup_llm(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel("gemini-1.5-flash")
            self.use_llm = True
            print("✓ QA Engine: using Gemini 1.5 Flash")
        else:
            self.use_llm = False
            print("⚠ QA Engine: using rule-based fallback (set GEMINI_API_KEY for LLM)")

    def answer(self, question: str, receipt_db_id: Optional[str] = None) -> Dict:
        """
        Answer a natural language question grounded in stored receipt data.
        """
        # 1. Retrieve relevant receipts
        if receipt_db_id:
            receipts = [self.store.get(receipt_db_id)]
            receipts = [r for r in receipts if r]
        else:
            # Use FTS to find relevant receipts
            keywords = self._extract_keywords(question)
            receipts = []
            for kw in keywords:
                receipts.extend(self.store.search(kw))
            if not receipts:
                receipts = self.store.get_all()[:5]   # fallback: most recent

        if not receipts:
            return {
                "question": question,
                "answer": "No receipts found in the database.",
                "source_receipts": [],
                "reasoning": "Database is empty.",
            }

        # 2. Build context
        context = self._build_context(receipts)

        # 3. Generate answer
        if self.use_llm:
            answer, reasoning = self._llm_answer(question, context)
        else:
            answer, reasoning = self._rule_based_answer(question, receipts)

        return {
            "question": question,
            "answer": answer,
            "source_receipts": [r.get("id", r.get("receipt_db_id", "")) or "" for r in receipts[:3]],
            "reasoning": reasoning,
        }

    def _build_context(self, receipts: List[Dict]) -> str:
        """
        Build a compact JSON context string from receipts.
        Only includes fields relevant to the extraction task.
        """
        simplified = []
        for r in receipts[:5]:    # limit to 5 receipts to stay within context window
            simplified.append({
                "id":         r.get("id", ""),
                "vendor":     r.get("vendor", "unknown"),
                "date":       r.get("date", "unknown"),
                "total":      r.get("total", "unknown"),
                "receipt_id": r.get("receipt_id", ""),
            })
        return json.dumps(simplified, indent=2)

    def _llm_answer(self, question: str, context: str):
        """
        Use Gemini Flash to answer, strictly grounded in context.
        """
        prompt = f"""You are a precise receipt data assistant.
Answer the user's question using ONLY the receipt data provided below.
Do not invent or infer any information not present in the data.
If the answer is not in the data, say "This information is not available in the stored receipts."

RECEIPT DATA:
{context}

QUESTION: {question}

Provide:
1. A direct answer (1-2 sentences)
2. Which receipt(s) you used to answer (by vendor or id)

Answer:"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()

            # Split answer from reasoning heuristically
            lines = text.split("\n")
            answer_lines    = [l for l in lines if not l.lower().startswith("receipt")]
            reasoning_lines = [l for l in lines if l.lower().startswith("receipt")]

            return (
                " ".join(answer_lines).strip() or text,
                " ".join(reasoning_lines).strip() or "Based on stored receipt data.",
            )
        except Exception as e:
            return (f"LLM error: {str(e)}", "")

    def _rule_based_answer(self, question: str, receipts: List[Dict]) -> tuple:
        """
        Simple keyword-matching fallback when no LLM is available.
        Covers the most common question types from the assessment spec.
        """
        q = question.lower()
        results = []

        if any(kw in q for kw in ["total", "amount", "price", "cost", "pay"]):
            for r in receipts:
                if r.get("total"):
                    results.append(f"{r.get('vendor','Receipt')}: {r['total']}")
            if results:
                return (
                    "Total amounts — " + "; ".join(results[:3]),
                    "Matched 'total/amount' keyword → returned total field."
                )

        if any(kw in q for kw in ["date", "when", "time"]):
            for r in receipts:
                if r.get("date"):
                    results.append(f"{r.get('vendor','Receipt')}: {r['date']}")
            if results:
                return (
                    "Dates — " + "; ".join(results[:3]),
                    "Matched 'date/when' keyword → returned date field."
                )

        if any(kw in q for kw in ["vendor", "store", "shop", "where", "merchant"]):
            vendors = [r["vendor"] for r in receipts if r.get("vendor")]
            if vendors:
                return (
                    "Vendors: " + ", ".join(set(vendors[:5])),
                    "Matched 'vendor/store' keyword → returned vendor field."
                )

        return (
            "Please ask about total amount, date, or vendor name.",
            "No matching keyword found. Try: 'What is the total?', 'What is the date?'",
        )

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract potential search keywords from the question."""
        # Remove stop words and short tokens
        stop = {"what", "is", "the", "a", "an", "of", "for", "in", "on", "at",
                "this", "that", "how", "much", "many", "which", "when", "where"}
        words = re.findall(r"\w+", question.lower())
        return [w for w in words if w not in stop and len(w) > 2]
