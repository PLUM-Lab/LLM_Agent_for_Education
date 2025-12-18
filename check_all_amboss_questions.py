#!/usr/bin/env python3
"""检查所有 Amboss 题目的图片分配情况"""

import json

# 读取文件
with open('qbank_amboss_openai.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("所有 Amboss 题目的图片分配情况")
print("=" * 80)

print(f"\n总题目数: {len(data)}\n")

for i, q in enumerate(data, 1):
    page = q.get('page_number', 'N/A')
    images = q.get('images', [])
    question_text = q.get('question', '')
    
    # 提取题目编号（如果有）
    import re
    match = re.match(r'^(\d+)\.', question_text)
    question_num = match.group(1) if match else '?'
    
    # 显示题目信息
    has_images = '✓' if images else '✗'
    print(f"题目 {i} (编号 {question_num}): page_number={page}, 图片={has_images}")
    
    if images:
        print(f"  图片路径: {images}")
    
    # 显示题目开头（前80字符）
    question_preview = question_text[:80].replace('\n', ' ')
    print(f"  题目: {question_preview}...")
    print()

print("\n" + "=" * 80)
print("图片分配可能的问题:")
print("=" * 80)
print("如果图片分配错误，可能的原因：")
print("1. OpenAI API 返回的 page_number 不准确")
print("2. 题目跨页，但只分配了第一页的图片")
print("3. 图片实际上属于其他题目")
print("\n请检查上述信息，找出图片分配错误的题目。")




