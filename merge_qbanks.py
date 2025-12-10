#!/usr/bin/env python3
"""
合并两个题库JSON文件
"""

import json
from pathlib import Path

def merge_qbanks(file1: str, file2: str, output_file: str):
    """
    合并两个题库文件
    
    Args:
        file1: 第一个文件路径
        file2: 第二个文件路径
        output_file: 输出文件路径
    """
    print(f"Reading {file1}...")
    with open(file1, 'r', encoding='utf-8') as f:
        questions1 = json.load(f)
    
    print(f"Reading {file2}...")
    with open(file2, 'r', encoding='utf-8') as f:
        questions2 = json.load(f)
    
    print(f"\nFile 1 ({file1}): {len(questions1)} questions")
    print(f"File 2 ({file2}): {len(questions2)} questions")
    
    # 合并题目：将 file2 (Amboss) 放在前面，file1 (Surgery) 放在后面
    combined_questions = questions2 + questions1
    
    print(f"\nCombined: {len(combined_questions)} questions")
    
    # 保存到新文件
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_questions, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Successfully merged {len(combined_questions)} questions to {output_file}")
    print(f"  Source files preserved: {file1}, {file2}")

if __name__ == '__main__':
    file1 = 'qbank_surgery_formatted.json'
    file2 = 'qbank_amboss_openai.json'
    output_file = 'qbank_combined.json'
    
    merge_qbanks(file1, file2, output_file)

