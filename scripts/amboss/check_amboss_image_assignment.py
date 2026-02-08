#!/usr/bin/env python3
"""检查 Amboss 题目的图片分配情况"""

import json

# 读取文件
with open('qbank_amboss_openai.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("检查 Amboss 题目的图片分配")
print("=" * 60)

print(f"\n总题目数: {len(data)}")

# 检查有图片的题目
questions_with_images = []
for i, q in enumerate(data, 1):
    page = q.get('page_number', 'N/A')
    images = q.get('images', [])
    if images:
        questions_with_images.append((i, page, images))

print(f"\n有图片的题目数: {len(questions_with_images)}")

if questions_with_images:
    print("\n有图片的题目详情:")
    for idx, page, images in questions_with_images:
        q = data[idx - 1]  # 索引从0开始
        print(f"\n题目 {idx}:")
        print(f"  page_number: {page}")
        print(f"  images: {images}")
        # 显示题目开头
        question_text = q.get('question', '')[:100]
        print(f"  题目开头: {question_text}...")

# 检查图片文件名和页码的对应关系
print("\n" + "=" * 60)
print("检查图片文件名和页码的对应关系")
print("=" * 60)

for idx, page, images in questions_with_images:
    for img_path in images:
        # 从图片路径提取页码信息
        # 例如: "images/Amboss Questions/page_8_img_1.png"
        if 'page_' in img_path:
            import re
            match = re.search(r'page_(\d+)_img', img_path)
            if match:
                img_page = int(match.group(1))
                if img_page != page:
                    print(f"\n⚠️  题目 {idx} 的图片分配可能有问题:")
                    print(f"  题目 page_number: {page}")
                    print(f"  图片文件名中的页码: {img_page}")
                    print(f"  图片路径: {img_path}")
                    print(f"  题目开头: {data[idx-1].get('question', '')[:80]}...")

