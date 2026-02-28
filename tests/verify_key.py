"""Quick script to verify OpenAI API key."""
import re
from pathlib import Path
path = Path(__file__).parent / 'api-key.js'
content = path.read_text(encoding='utf-8')
m = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
key = m.group(1) if m else None
if not key:
    print('No key found in api-key.js')
    exit(1)
masked = key[:8] + '...' + key[-4:] if len(key) > 12 else '***'
print(f'Key: {masked} (len={len(key)})')
from openai import OpenAI
try:
    c = OpenAI(api_key=key)
    list(c.models.list())
    print('OK: API key is valid')
except Exception as e:
    print('FAIL:', str(e))
