"""
从 Word 文档提取领域结构并转换为 JSON 格式
Extract domain structure from Word document and convert to JSON format

这个脚本将读取 Word 文档，提取领域、主题和子主题的结构，
然后保存为 domain_question_generator.py 可以使用的 JSON 格式。
"""
import json
from pathlib import Path
import re

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("[ERROR] python-docx not installed. Please install it with: pip install python-docx")


def extract_text_from_docx(docx_path: str) -> str:
    """从 Word 文档提取纯文本"""
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)


def parse_domains_from_text(text: str) -> dict:
    """
    从文本中解析领域结构
    假设格式：
    Domain Name
        Topic Name
            - Subtopic 1
            - Subtopic 2
    """
    domains = []
    lines = text.split("\n")
    
    current_domain = None
    current_topic = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # 检查是否是领域名称（通常是大标题，可能是加粗或大写）
        # 检查是否是主题名称（可能是缩进的）
        # 检查是否是子主题（通常以 - 或 • 开头，或者数字编号）
        
        # 如果行首没有缩进或很少缩进，可能是领域或主题
        # 如果行首有很多空格或以 - • 1. 等开头，可能是子主题
        
        # 简单启发式：尝试识别层级结构
        # 让我们使用更智能的方法：让LLM帮助我们解析
        
        i += 1
    
    return {"domains": domains}


def parse_with_llm(text: str) -> dict:
    """
    使用 OpenAI API 解析 Word 文档文本，提取领域结构
    """
    try:
        from openai import OpenAI
        import os
        import re
        
        # 获取 API 密钥
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            config_path = Path(__file__).resolve().parent.parent / "api-key.js"
            if config_path.exists():
                content = config_path.read_text(encoding="utf-8")
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
        
        if not api_key:
            raise RuntimeError("OpenAI API key not found")
        
        client = OpenAI(api_key=api_key)
        
        # Limit text to avoid token limits
        text_to_parse = text[:12000] if len(text) > 12000 else text
        
        prompt = f"""Parse the following text from a Surgery Clerkship Knowledge Domains document and extract the domain/topic/subtopic structure.

The document structure is:
- Major headings (all caps or bold) are DOMAINS
- Items directly under domains are TOPICS
- Items nested under topics are SUBTOPICS

Return ONLY valid JSON in this format:
{{
  "domains": [
    {{
      "name": "Domain Name",
      "level": "MS3",
      "topics": [
        {{
          "name": "Topic Name",
          "subtopics": [
            "Subtopic 1",
            "Subtopic 2"
          ]
        }}
      ]
    }}
  ]
}}

Text to parse:
{text_to_parse}

Extract the hierarchical structure: Domains → Topics → Subtopics.
Make sure to set level as "MS3" for all domains.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured information from documents. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=16000  # 增加max_tokens以确保完整的JSON结构能够返回（16个domains需要大量tokens）
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to parse with LLM: {e}")
        raise


def main():
    docx_path = Path("Surgery Clerkship Knowledge Domains.docx")
    output_path = Path("surgery_domains_with_subtopics.json")
    
    if not docx_path.exists():
        print(f"[ERROR] File not found: {docx_path}")
        return
    
    print(f"Reading Word document: {docx_path}")
    
    if not HAS_DOCX:
        print("[ERROR] Please install python-docx: pip install python-docx")
        return
    
    # 提取文本
    text = extract_text_from_docx(str(docx_path))
    print(f"Extracted {len(text)} characters from document")
    
    # 使用 LLM 解析结构
    print("Parsing structure with LLM...")
    try:
        domains_data = parse_with_llm(text)
    except Exception as e:
        print(f"[ERROR] Failed to parse: {e}")
        print("\nTrying manual parsing instead...")
        # 如果 LLM 解析失败，可以尝试手动解析
        domains_data = parse_domains_from_text(text)
    
    # 保存 JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(domains_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved domain structure to: {output_path}")
    
    # 统计信息
    domains = domains_data.get("domains", [])
    total_topics = sum(len(d.get("topics", [])) for d in domains)
    total_subtopics = sum(
        len(t.get("subtopics", []))
        for d in domains
        for t in d.get("topics", [])
    )
    
    print(f"\nStatistics:")
    print(f"  Domains: {len(domains)}")
    print(f"  Topics: {total_topics}")
    print(f"  Subtopics: {total_subtopics}")
    
    print(f"\nNext step: Run domain_question_generator.py with this config file")


if __name__ == "__main__":
    main()

