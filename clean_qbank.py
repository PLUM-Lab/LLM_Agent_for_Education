"""
清理 qbank_questions.json 中的冗余内容
- 删除选项少于2个的题目
- 删除选项不完整的题目
- 删除题目文本为空的题目
"""

import json
from pathlib import Path

def clean_qbank_json(input_file: str = "qbank_questions.json", output_file: str = "qbank_questions.json"):
    """清理 Qbank JSON 文件，删除冗余和无效的题目"""
    
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"原始题目数量: {len(questions)}")
    
    # 清理条件
    cleaned_questions = []
    removed_count = 0
    
    for i, q in enumerate(questions):
        # 检查条件
        options = q.get('options', {})
        question_text = q.get('question', '').strip()
        
        # 条件1：必须有题目文本
        if not question_text or len(question_text) < 10:
            print(f"删除题目 {i+1}: 题目文本为空或太短")
            removed_count += 1
            continue
        
        # 条件2：必须有至少2个选项
        if not options or len(options) < 2:
            print(f"删除题目 {i+1}: 选项数量不足（当前: {len(options)}）")
            removed_count += 1
            continue
        
        # 条件3：选项不能为空
        valid_options = {k: v for k, v in options.items() if v and len(v.strip()) > 0}
        if len(valid_options) < 2:
            print(f"删除题目 {i+1}: 有效选项数量不足（当前: {len(valid_options)}）")
            removed_count += 1
            continue
        
        # 条件4：正确答案必须在选项中
        correct_answer = q.get('correct_answer', '')
        if correct_answer and correct_answer not in valid_options:
            print(f"删除题目 {i+1}: 正确答案 '{correct_answer}' 不在选项中")
            removed_count += 1
            continue
        
        # 更新选项（只保留有效选项）
        q['options'] = valid_options
        
        # 如果正确答案为空，尝试从选项中推断（选择第一个）
        if not correct_answer and valid_options:
            q['correct_answer'] = list(valid_options.keys())[0]
        
        cleaned_questions.append(q)
    
    print(f"\n清理结果:")
    print(f"  保留题目: {len(cleaned_questions)}")
    print(f"  删除题目: {removed_count}")
    
    # 保存清理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_questions, f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存到: {output_file}")
    
    return cleaned_questions

if __name__ == "__main__":
    clean_qbank_json()

