#!/bin/bash
# 用 curl 测试 OpenAI API 密钥
# 用法：bash test_api_curl.sh  或  ./test_api_curl.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KEY=""
if [ -f "$PROJECT_ROOT/api-key.js" ]; then
    KEY=$(grep -oP "OPENAI_API_KEY\s*[=:]\s*['\"]\K[^'\"]+" "$PROJECT_ROOT/api-key.js" 2>/dev/null || \
          sed -n "s/.*OPENAI_API_KEY[=:][[:space:]]*['\"]\\([^'\"]*\\)['\"].*/\\1/p" "$PROJECT_ROOT/api-key.js")
fi
[ -z "$KEY" ] && KEY="${OPENAI_API_KEY}"

if [ -z "$KEY" ]; then
    echo "[失败] 未找到 API 密钥（api-key.js 或 OPENAI_API_KEY）"
    exit 1
fi

echo "测试 OpenAI API..."
RESP=$(curl -s -X POST "https://api.openai.com/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say OK"}],"max_tokens":10}')

if echo "$RESP" | grep -q '"choices"'; then
    echo "[成功] $(echo "$RESP" | grep -oP '"content":"\K[^"]*' | head -1)"
    exit 0
else
    echo "[失败] $RESP"
    echo "$RESP" | grep -q "billing_not_active" && echo "  → 请到 https://platform.openai.com/account/billing 完成账单设置"
    exit 1
fi
