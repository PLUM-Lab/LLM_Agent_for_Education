"""Test evaluate_student_thinking endpoint and capture full error."""
import requests
import json

url = "http://127.0.0.1:5000/evaluate_student_thinking"
payload = {
    "question": "A 25-year-old man presents with a deep laceration. Which is the first step in wound healing?",
    "choices": {
        "A": "Collagen deposition",
        "B": "Epithelialization",
        "C": "Hemostasis",
        "D": "Inflammation",
    },
    "student_answer": "A",
    "correct_answer": "C",
    "student_thinking": "I thought collagen deposition comes first",
}

try:
    r = requests.post(url, json=payload, timeout=60)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:1000]}")
except Exception as e:
    print(f"Request failed: {e}")
