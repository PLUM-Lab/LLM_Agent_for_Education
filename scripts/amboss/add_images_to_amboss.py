#!/usr/bin/env python3
"""
为现有的 Amboss 题目文件添加图片路径

从 images/Amboss Questions/ 目录中查找图片，根据 page_number 匹配并添加到题目数据中。
"""

import json
from pathlib import Path
import re

def find_images_for_page(page_num: int, images_dir: Path) -> list:
    """根据页码查找对应的图片文件"""
    images = []
    
    # 查找匹配的图片文件
    # 格式：page_{page_num}_img_{index}.png
    pattern = re.compile(rf'page_{page_num}_img_(\d+)\.(png|jpg|jpeg)', re.IGNORECASE)
    
    for img_file in images_dir.iterdir():
        if img_file.is_file():
            match = pattern.match(img_file.name)
            if match:
                # 使用相对路径，从项目根目录开始
                relative_path = f"images/Amboss Questions/{img_file.name}"
                images.append(relative_path)
    
    # 如果没找到，尝试旧格式：{pdf_name}_{page_num}_{index}.png
    if not images:
        pattern_old = re.compile(rf'Amboss Questions_{page_num}_(\d+)\.(png|jpg|jpeg)', re.IGNORECASE)
        for img_file in images_dir.parent.iterdir():
            if img_file.is_file() and img_file.name.startswith('Amboss Questions_'):
                match = pattern_old.match(img_file.name)
                if match:
                    relative_path = f"images/{img_file.name}"
                    images.append(relative_path)
    
    return sorted(images)

def add_images_to_questions(questions_file: str = 'qbank_amboss_openai.json', 
                            images_dir: str = 'images/Amboss Questions'):
    """为题目添加图片路径"""
    
    # 读取题目文件
    questions_path = Path(questions_file)
    if not questions_path.exists():
        print(f"错误：找不到题目文件 {questions_file}")
        return
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"已加载 {len(questions)} 道题目")
    
    # 检查图片目录
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"警告：图片目录不存在 {images_dir}")
        print("将创建空目录...")
        images_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    updated_count = 0
    total_images = 0
    
    # 为每道题目添加图片
    for i, question in enumerate(questions):
        page_num = question.get('page_number')
        if not page_num:
            continue
        
        # 查找该页的图片
        images = find_images_for_page(page_num, images_path)
        
        if images:
            question['images'] = images
            updated_count += 1
            total_images += len(images)
            print(f"题目 {i+1} (页码 {page_num}): 添加了 {len(images)} 张图片")
        else:
            # 即使没有图片，也添加空数组，保持数据结构一致
            if 'images' not in question:
                question['images'] = []
    
    # 保存更新后的题目
    output_file = questions_path
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！")
    print(f"  - 更新了 {updated_count} 道题目")
    print(f"  - 共添加了 {total_images} 张图片")
    print(f"  - 已保存到 {output_file}")

if __name__ == '__main__':
    add_images_to_questions()

