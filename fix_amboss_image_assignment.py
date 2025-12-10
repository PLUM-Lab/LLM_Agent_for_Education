#!/usr/bin/env python3
"""
修复 Amboss 题目的图片分配

使用方法：
1. 编辑此文件，在 image_assignments 字典中指定正确的图片分配
2. 格式：{题目索引（从0开始）: [图片路径列表]}
3. 运行：python fix_amboss_image_assignment.py
"""

import json
from pathlib import Path

# 读取原始文件
input_file = 'qbank_amboss_openai.json'
output_file = 'qbank_amboss_openai.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("修复 Amboss 题目的图片分配")
print("=" * 80)

# ============================================================================
# 在这里指定正确的图片分配
# 格式：{题目索引（从0开始）: [图片路径列表]}
# 例如：{7: ['images/Amboss Questions/page_8_img_1.png']}
# ============================================================================
# 
# 当前图片分配：
# - 题目 8 (索引 7): page_number=8, 图片=page_8_img_1.png
# - 题目 17 (索引 16): page_number=19, 图片=page_19_img_1.png
#
# 如果图片分配错误，请在这里修改：
# image_assignments = {
#     7: ['images/Amboss Questions/page_8_img_1.png'],  # 题目 8
#     16: ['images/Amboss Questions/page_19_img_1.png'],  # 题目 17
# }
# ============================================================================

# 默认：保持当前分配（不做修改）
image_assignments = {}

# 如果需要修改，请取消注释并编辑：
# image_assignments = {
#     # 示例：将题目 8 的图片改为其他图片
#     # 7: ['images/Amboss Questions/page_19_img_1.png'],
#     # 示例：移除题目 17 的图片
#     # 16: [],
# }

# 应用图片分配
modified = False
for idx, images in image_assignments.items():
    if 0 <= idx < len(data):
        old_images = data[idx].get('images', [])
        data[idx]['images'] = images
        modified = True
        print(f"\n题目 {idx + 1} (索引 {idx}):")
        print(f"  旧图片: {old_images}")
        print(f"  新图片: {images}")
    else:
        print(f"\n⚠️  警告: 索引 {idx} 超出范围（总题目数: {len(data)}）")

if not modified:
    print("\n没有指定图片分配修改，文件保持不变。")
    print("如需修改，请编辑此脚本中的 image_assignments 字典。")
else:
    # 保存修改后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已保存修改到 {output_file}")
    print("\n下一步：")
    print("1. 运行 merge_qbanks.py 更新 qbank_combined.json")
    print("2. 刷新浏览器页面查看效果")

