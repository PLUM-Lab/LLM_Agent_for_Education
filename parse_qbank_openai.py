"""
完全使用 OpenAI Vision API 解析 Qbank PDF 文件

功能：
    使用 OpenAI GPT-4o Vision API 完全解析 PDF，不依赖传统正则表达式方法

优势：
    - 更智能：AI 能理解 PDF 的结构和内容
    - 更准确：可能比正则表达式更准确
    - 处理复杂格式：能处理各种 PDF 格式

缺点：
    - 成本：需要 API 调用费用
    - 速度：API 调用可能比本地解析慢
    - 限制：有速率限制

使用方法：
    python parse_qbank_openai.py --dir "Qbanks and Practice Exams" -o qbank_questions_openai.json
    python parse_qbank_openai.py "Surgery 3.pdf" -o output.json
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

def get_openai_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """
    获取 OpenAI API 密钥。
    
    优先级：
        1. 函数参数
        2. 环境变量 OPENAI_API_KEY
        3. api-key.js 文件
    
    Returns:
        str: API 密钥，如果未找到返回 None
    """
    if api_key:
        return api_key
    
    # 尝试从环境变量读取
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # 尝试从 api-key.js 读取
    api_key_path = Path('api-key.js')
    if api_key_path.exists():
        try:
            with open(api_key_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r"OPENAI_API_KEY\s*=\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    return match.group(1)
        except Exception as e:
            print(f"Warning: Failed to read API key from api-key.js: {e}")
    
    return None


def parse_pdf_with_openai(pdf_path: str, api_key: Optional[str] = None, model: str = "gpt-4o") -> List[Dict]:
    """
    使用 OpenAI Vision API 完全解析 PDF 文件。
    
    Args:
        pdf_path: PDF 文件路径
        api_key: OpenAI API 密钥（如果为 None，尝试从环境变量或 api-key.js 读取）
        model: 使用的 OpenAI 模型（默认：gpt-4o，支持视觉）
    
    Returns:
        List[Dict]: 题目列表
    """
    try:
        from openai import OpenAI
        import base64
    except ImportError:
        print("Error: OpenAI library not installed. Run: pip install openai")
        return []
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Error: PyMuPDF not installed. Run: pip install pymupdf")
        return []
    
    # 获取 API 密钥
    api_key = get_openai_api_key(api_key)
    if not api_key:
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide api_key parameter.")
        return []
    
    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=api_key)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Error opening PDF: {e}")
        return []
    
    print(f"  Processing {len(doc)} pages with OpenAI Vision API...")
    
    questions = []
    
    # 逐页处理
    for page_num, page in enumerate(doc, start=1):
        try:
            print(f"    Processing page {page_num}/{len(doc)}...", end=" ")
            
            # 将页面转换为图片（PNG 格式）
            # 使用 2x 缩放以提高清晰度
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            # 将图片转换为 base64 编码
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # 构建提示词
            system_prompt = """你是一个专业的医学考试题目解析专家。请分析这张 PDF 页面图片，提取所有题目信息。

对于每个题目，请提取以下信息：
1. question: 题目文本（包括题干和 Tip，如果有 Tip 请包含在题干中）
2. options: 选项字典，格式为 {"A": "选项A文本", "B": "选项B文本", ...}（支持 A-J，最多 10 个选项）
3. correct_answer: 正确答案字母（A-J）
4. tip: 提示文本（如果有，但应该已经包含在题干中，这里可以为空字符串）
5. explanation: 解释文本（完整提取，保留段落结构，使用 \\n\\n 分隔段落）

输出格式：必须是一个 JSON 对象，包含一个 "questions" 字段，值为题目数组。
格式示例：
{
  "questions": [
    {
      "question": "题目文本...",
      "options": {"A": "选项A", "B": "选项B"},
      "correct_answer": "A",
      "tip": "",
      "explanation": "解释文本..."
    }
  ]
}

如果页面中有多个题目，请提取所有题目。
如果页面中没有题目，返回 {"questions": []}。

重要要求：
- 确保提取完整的题目文本，包括所有实验数据（Laboratory studies show: 等）
- Tip 内容应该包含在题干中（在 question 字段中）
- Explanation 要完整提取，保留段落结构（使用 \\n\\n 分隔段落）
- 选项要准确提取，不要包含解释文本
- 选项应该简洁，通常不超过 200 个字符
- 如果题目包含图片，在 question 中说明"""
            
            user_prompt = f"请分析这张 PDF 页面（第 {page_num} 页），提取所有题目信息。严格按照要求的 JSON 格式输出。"
            
            # 调用 OpenAI Vision API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # 低温度以确保准确性
            )
            
            # 解析响应
            response_text = response.choices[0].message.content
            
            try:
                # 尝试解析 JSON
                result = json.loads(response_text)
                
                # 提取题目列表
                if isinstance(result, dict):
                    # 如果返回的是对象，尝试找到题目数组
                    if 'questions' in result:
                        page_questions = result['questions']
                    elif 'items' in result:
                        page_questions = result['items']
                    elif 'data' in result:
                        page_questions = result['data']
                    else:
                        # 尝试直接使用值（如果只有一个键）
                        values = list(result.values())
                        if len(values) == 1 and isinstance(values[0], list):
                            page_questions = values[0]
                        else:
                            page_questions = []
                elif isinstance(result, list):
                    page_questions = result
                else:
                    page_questions = []
                
                # 处理每个题目
                for q in page_questions:
                    if isinstance(q, dict):
                        # 添加元数据
                        q['source'] = "OpenAI Vision API"
                        q['source_file'] = Path(pdf_path).name
                        q['page_number'] = page_num
                        
                        # 确保字段存在
                        if 'question' not in q:
                            continue
                        if 'options' not in q:
                            q['options'] = {}
                        if 'correct_answer' not in q:
                            q['correct_answer'] = 'A'
                        if 'tip' not in q:
                            q['tip'] = ""
                        if 'explanation' not in q:
                            q['explanation'] = ""
                        
                        questions.append(q)
                
                print(f"✓ Extracted {len(page_questions)} question(s)")
                
            except json.JSONDecodeError as e:
                print(f"✗ Failed to parse JSON: {e}")
                print(f"    Response preview: {response_text[:200]}...")
                continue
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    doc.close()
    return questions


def parse_directory(directory: str, output_file: str = 'qbank_questions_openai.json',
                   api_key: Optional[str] = None) -> List[Dict]:
    """
    批量解析目录下的所有 PDF 文件（完全使用 OpenAI）。
    
    Args:
        directory: PDF 文件目录路径
        output_file: 输出 JSON 文件路径
        api_key: OpenAI API 密钥
    
    Returns:
        List[Dict]: 所有题目的合并列表
    """
    all_questions = []
    
    # 查找所有 PDF 文件
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {directory}")
    print()
    
    # 逐个处理每个 PDF 文件
    for pdf_file in sorted(pdf_files):
        # 过滤：跳过只有题目没有答案的文件
        if 'Questions' in pdf_file.name and 'Answers' not in pdf_file.name:
            if 'amboss' not in pdf_file.name.lower():
                print(f"Skipping (no answers): {pdf_file.name}")
                continue
        
        # 解析当前文件
        print(f"Processing: {pdf_file.name}")
        questions = parse_pdf_with_openai(str(pdf_file), api_key)
        print(f"  Parsed: {len(questions)} questions")
        all_questions.extend(questions)
        print()
    
    # 保存到 JSON 文件
    if all_questions and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_questions)} questions to {output_file}")
    
    return all_questions


def main():
    """
    命令行入口函数。
    """
    parser = argparse.ArgumentParser(
        description='Qbank PDF Parser (OpenAI Only) - 完全使用 OpenAI Vision API 解析医学考试题库 PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 解析单个 PDF
  python parse_qbank_openai.py "Surgery 3 - Answers.pdf"
  
  # 解析目录下所有 PDF
  python parse_qbank_openai.py --dir "Qbanks and Practice Exams"
  
  # 指定输出文件
  python parse_qbank_openai.py --dir "Qbanks and Practice Exams" -o my_questions.json
  
  # 指定 API 密钥
  python parse_qbank_openai.py --dir "Qbanks and Practice Exams" --api-key sk-...
        """
    )
    
    # 位置参数：PDF 文件路径（可选）
    parser.add_argument('pdf', nargs='?', help='PDF 文件路径（单个文件）')
    
    # 可选参数：目录
    parser.add_argument('--dir', '-d', help='PDF 文件目录（批量解析）')
    
    # 可选参数：输出文件
    parser.add_argument('--output', '-o', default='qbank_questions_openai.json', 
                       help='输出 JSON 文件路径（默认：qbank_questions_openai.json）')
    
    # 可选参数：OpenAI API 密钥
    parser.add_argument('--api-key', '--openai-key',
                       help='OpenAI API 密钥（如果不提供，尝试从环境变量或 api-key.js 读取）')
    
    # 可选参数：OpenAI 模型
    parser.add_argument('--model', '-m', default='gpt-4o',
                       help='OpenAI 模型（默认：gpt-4o）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 显示标题
    print("=" * 60)
    print("Qbank PDF Parser (OpenAI Only)")
    print("=" * 60)
    print()
    
    # 根据参数选择处理方式
    if args.dir:
        # 情况 1：指定了目录，批量解析
        parse_directory(args.dir, args.output, args.api_key)
    elif args.pdf:
        # 情况 2：指定了单个文件，解析单个文件
        questions = parse_pdf_with_openai(args.pdf, args.api_key, args.model)
        if questions:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"\nSaved {len(questions)} questions to {args.output}")
    else:
        # 情况 3：没有指定参数，尝试默认目录
        qbank_dir = "Qbanks and Practice Exams"
        if os.path.exists(qbank_dir):
            parse_directory(qbank_dir, args.output, args.api_key)
        else:
            # 默认目录不存在，显示帮助信息
            parser.print_help()


if __name__ == "__main__":
    main()

