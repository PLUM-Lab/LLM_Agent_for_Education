#!/usr/bin/env python3
"""
测试脚本：演示 understanding_level 从 message.content 的打印输出
"""

import requests
import json
import time

def test_understanding_level():
    """测试 understanding_level 打印功能"""
    
    print("=" * 60)
    print("Test understanding_level printing functionality")
    print("=" * 60)
    print()
    
    # Test data
    test_data = {
        "question": "What is the primary treatment for acute appendicitis?",
        "choices": {
            "A": "Immediate surgical appendectomy",
            "B": "Antibiotic therapy alone",
            "C": "Observation and pain management",
            "D": "Laparoscopic exploration only"
        },
        "student_answer": "B",  # Wrong answer
        "correct_answer": "A",  # Correct answer
        "student_thinking": "I thought antibiotics would be sufficient to treat the infection"
    }
    
    print("Sending test request to /evaluate_student_thinking...")
    print(f"Student thinking: {test_data['student_thinking']}")
    print()
    print("Please check console output, you should see:")
    print("  [DEBUG] understanding_level from message.content: <level>")
    print()
    print("-" * 60)
    
    try:
        # 发送请求到 RAG 服务器
        response = requests.post(
            'http://localhost:5000/evaluate_student_thinking',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print()
            print("[SUCCESS] Request successful!")
            print()
            print("Response result:")
            print(f"  understanding_level: {result.get('understanding_level')}")
            print(f"  action_type: {result.get('action_type')}")
            print(f"  reasoning: {result.get('reasoning', '')[:100]}...")
            print()
            print("Please check the RAG server console, you should see:")
            print("  [DEBUG] understanding_level from message.content: <level>")
        else:
            print(f"[ERROR] Request failed: HTTP {response.status_code}")
            print(f"  Error message: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print()
        print("[ERROR] Cannot connect to server (http://localhost:5000)")
        print()
        print("Please start the RAG server first:")
        print("  python start.py --rag")
        print("  or")
        print("  python start.py  # Start all services")
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_understanding_level()
