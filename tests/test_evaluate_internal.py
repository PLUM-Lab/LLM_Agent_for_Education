"""Test evaluate_student_thinking internally (no server needed) to capture error."""
import sys
sys.path.insert(0, '.')

# Trigger RAG init first
import os
os.environ.setdefault('REQUIRE_RERANKER', '0')  # Skip reranker for quick test

from rag_server import app

payload = {
    "question": "A 25-year-old man presents with a deep laceration. Which is the first step in wound healing?",
    "choices": {"A": "Collagen deposition", "B": "Epithelialization", "C": "Hemostasis", "D": "Inflammation"},
    "student_answer": "A",
    "correct_answer": "C",
    "student_thinking": "I thought collagen deposition comes first",
}

with app.test_client() as c:
    r = c.post("/evaluate_student_thinking", json=payload)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.get_data(as_text=True)[:1500]}")
