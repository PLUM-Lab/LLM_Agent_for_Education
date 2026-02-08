#!/usr/bin/env python3
"""
测试 OpenAI API 密钥是否有效
用法：python test_api_key.py
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_api_key():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        config_path = PROJECT_ROOT / 'api-key.js'
        if config_path.exists():
            try:
                content = config_path.read_text(encoding='utf-8')
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
            except Exception as e:
                print(f"[!] 读取 api-key.js 失败: {e}")
    return (api_key or "").strip()

def main():
    print("=" * 50)
    print("OpenAI API 密钥测试")
    print("=" * 50)
    
    api_key = get_api_key()
    if not api_key:
        print("\n[失败] 未找到 API 密钥")
        print("  请创建 api-key.js 或设置环境变量 OPENAI_API_KEY")
        return 1
    
    print(f"\n[OK] 已加载密钥 (前缀: {api_key[:10]}...)")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("\n正在调用 OpenAI API（简单测试）...")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OK' in one word."}],
            max_tokens=10
        )
        reply = resp.choices[0].message.content
        print(f"\n[成功] API 响应: {reply.strip()}")
        print("\nAPI 密钥有效，Tutor 应可正常工作。")
        return 0
    except Exception as e:
        err = str(e).lower()
        print(f"\n[失败] {e}")
        if "api" in err and ("key" in err or "invalid" in err or "incorrect" in err):
            print("  原因: API 密钥无效或已过期")
        elif "billing" in err or "not active" in err:
            print("  原因: OpenAI 账户未激活，请到 https://platform.openai.com/account/billing 完成账单设置")
        elif "quota" in err or "insufficient" in err:
            print("  原因: 额度不足")
        elif "rate" in err:
            print("  原因: 请求过频")
        else:
            print("  请检查网络或 OpenAI 服务状态")
        return 1

if __name__ == "__main__":
    exit(main())
