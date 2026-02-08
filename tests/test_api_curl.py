#!/usr/bin/env python3
"""Test OpenAI API key using curl (reads from api-key.js)."""
import re
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "api-key.js", encoding="utf-8") as f:
    m = re.search(r"OPENAI_API_KEY\s*=\s*['\"]([^'\"]+)['\"]", f.read())
key = m.group(1) if m else ""
if not key:
    print("[FAIL] No key found in api-key.js")
    exit(1)

print("Key prefix:", key[:20] + "...")
result = subprocess.run(
    [
        "curl.exe", "-s", "-w", "\n%{http_code}",
        "-X", "POST", "https://api.openai.com/v1/chat/completions",
        "-H", "Authorization: Bearer " + key,
        "-H", "Content-Type: application/json",
        "-d", '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say OK"}],"max_tokens":10}',
    ],
    capture_output=True,
    text=True,
)
out = result.stdout
parts = out.rsplit("\n", 1)
body = parts[0]
code = parts[1] if len(parts) > 1 else ""
print("HTTP Status:", code)
if code == "200":
    d = json.loads(body)
    msg = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("[OK] API Key valid. Response:", msg)
else:
    print("[FAIL]")
    try:
        err = json.loads(body).get("error", {})
        print("Error:", err.get("message", body[:200]))
    except Exception:
        print(body[:300])
