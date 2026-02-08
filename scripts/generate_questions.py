"""
================================================================================
医学选择题生成器 - 使用 OpenAI API
================================================================================

功能：
    - 直接从 PDF 文件生成题目（无需转换为 TXT）
    - 为每个 PDF 文档生成指定数量的题目
    - 从每个文档的所有分块中随机选择生成题目
    - 使用 OpenAI GPT 模型生成高质量选择题

使用方法：
    1. 安装依赖：pip install openai langchain-community pypdf
    2. 配置 API 密钥（api-key.js 或环境变量）
    3. 将 PDF 文件放入 "Clinical Guidelines" 目录
    4. 运行：python generate_questions.py

输出：
    - questions.json：包含所有题目的 JSON 文件
    - questions.txt：人类可读的文本格式

作者：AI for Education (Wenchao Qin)
================================================================================
"""

# ============================================================================
# 导入依赖
# ============================================================================

from pathlib import Path
import json
import random
from typing import List, Dict
import time
import os
import sys

from openai import OpenAI

# LangChain 文档处理
try:
    from langchain_core.documents import Document as LangchainDocument
except ImportError:
    try:
        from langchain.docstore.document import Document as LangchainDocument
    except ImportError:
        from langchain.schema import Document as LangchainDocument

# 文本分割器
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

# 分词器（可选）
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    AutoTokenizer = None
    HAS_TOKENIZER = False

# PDF 加载器
try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF_LOADER = True
except ImportError:
    HAS_PDF_LOADER = False


# ============================================================================
# 依赖检查
# ============================================================================

def ensure_pdf_dependencies() -> bool:
    """确保 PDF 处理依赖已安装"""
    global HAS_PDF_LOADER
    
    if HAS_PDF_LOADER:
        return True
    
    print("Installing PDF processing dependencies...")
    import subprocess
    packages = ["langchain-community", "pypdf"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF_LOADER = True
    return True


# ============================================================================
# 辅助函数
# ============================================================================

def get_api_key() -> str:
    """获取 OpenAI API 密钥"""
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        config_path = Path(__file__).resolve().parent.parent / 'api-key.js'
        if config_path.exists():
            try:
                content = config_path.read_text()
                import re
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
                    print("[OK] API key loaded from api-key.js")
            except:
                pass
    
    if not api_key:
        print("\n" + "=" * 60)
        print("OpenAI API Key Required")
        print("=" * 60)
        api_key = input("\nEnter your OpenAI API key: ").strip()
    
    return api_key


def format_time(seconds: float) -> str:
    """将秒数格式化为人类可读的字符串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ============================================================================
# PDF 加载（按文档分组）
# ============================================================================

def load_pdfs_grouped(directory: str) -> Dict[str, List[LangchainDocument]]:
    """
    从目录加载所有 PDF 文件，按文件名分组
    
    参数：
        directory: 包含 PDF 文件的目录
    
    返回：
        字典，将文件名映射到 Document 对象列表
        
    示例：
        {
            "Guidelines for AAA repair.pdf": [doc1, doc2, doc3, ...],
            "Management of Acute MI.pdf": [doc1, doc2, ...],
            ...
        }
    """
    if not HAS_PDF_LOADER:
        ensure_pdf_dependencies()
    
    from langchain_community.document_loaders import PyPDFLoader
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"[ERROR] Directory does not exist: {directory}")
        return {}
    
    pdf_files = list(dir_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDF files found in {directory}")
        return {}
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Store documents grouped by file
    documents_by_file = {}
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Update metadata
            for doc in docs:
                doc.metadata['source'] = pdf_file.name
            
            documents_by_file[pdf_file.name] = docs
            print(f"  [OK] {pdf_file.name}: {len(docs)} pages")
            
        except Exception as e:
            print(f"  [X] {pdf_file.name}: {e}")
    
    total_pages = sum(len(docs) for docs in documents_by_file.values())
    print(f"Total: {len(documents_by_file)} files, {total_pages} pages loaded")
    
    return documents_by_file


# ============================================================================
# 文档分割
# ============================================================================

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n",
    "\n\n", "\n", " ", "",
]

EMBEDDING_MODEL_NAME = "thenlper/gte-small"


def split_documents(chunk_size: int, documents: List[LangchainDocument]) -> List[LangchainDocument]:
    """
    将文档分割成较小的块
    
    参数：
        chunk_size: 每块的最大 token 数
        documents: 要分割的文档列表
    
    返回：
        文档块列表（已去重）
    """
    if not HAS_TOKENIZER:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=int(chunk_size * 4 / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
    
    docs_processed = []
    for doc in documents:
        docs_processed += text_splitter.split_documents([doc])
    
    # Deduplicate
    unique_texts = {}
    docs_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_unique.append(doc)
    
    return docs_unique


# ============================================================================
# 题目生成（按文档）
# ============================================================================

def generate_questions_for_document(
    document_name: str,
    documents: List[LangchainDocument],
    num_questions: int,
    client,
    model: str,
    chunk_size: int
) -> List[Dict]:
    """
    为单个文档生成选择题
    
    参数：
        document_name: 文档名称
        documents: 文档的所有页面
        num_questions: 要生成的题目数量
        client: OpenAI 客户端
        model: 模型名称
        chunk_size: 块大小（token 数）
    
    返回：
        该文档生成的题目列表
    """
    # Split document into chunks
    chunks = split_documents(chunk_size=chunk_size, documents=documents)
    
    if not chunks:
        print(f"  [!] {document_name}: Cannot split, skipping")
        return []
    
    # Randomly select chunks
    if len(chunks) < num_questions:
        print(f"  [!] {document_name}: Only {len(chunks)} chunks available, using all")
        selected_chunks = chunks
    else:
        selected_chunks = random.sample(chunks, num_questions)
    
    print(f"  Selected {len(selected_chunks)} chunks from {len(chunks)} total")
    
    questions = []
    
    system_message = """You are a medical education expert who creates high-quality multiple choice questions from medical texts.

Generate a single multiple choice question that:
1. Tests important medical knowledge from the provided text
2. Has exactly 4 options (A, B, C, D)
3. Has only one correct answer
4. Has plausible but clearly incorrect distractors
5. Provides explanation for EACH option (why correct or why incorrect)

Return ONLY valid JSON:
{
  "question": "Question text here?",
  "options": {
    "A": "First option",
    "B": "Second option", 
    "C": "Third option",
    "D": "Fourth option"
  },
  "correct_answer": "A",
  "explanations": {
    "A": "Correct. Explanation why A is the right answer...",
    "B": "Incorrect. Explanation why B is wrong...",
    "C": "Incorrect. Explanation why C is wrong...",
    "D": "Incorrect. Explanation why D is wrong..."
  }
}"""
    
    for i, doc in enumerate(selected_chunks):
        print(f"    [{i+1}/{len(selected_chunks)}] Generating...", end=" ")
        
        user_content = f"""Based on this medical text, generate 1 multiple choice question:

---
{doc.page_content}
---

Return ONLY valid JSON."""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse
            question = None
            try:
                question = json.loads(response_text)
            except json.JSONDecodeError:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        question = json.loads(response_text[start_idx:end_idx])
                    except:
                        pass
            
            if question and isinstance(question, dict):
                question['source'] = document_name
                # Save the source chunk text for RAG context
                question['source_chunk'] = doc.page_content
                # Save page number from document metadata
                question['source_page'] = doc.metadata.get('page', 0) + 1  # PyPDFLoader uses 0-indexed pages
                questions.append(question)
                print("[OK]")
            else:
                print("[!] Parse failed")
                
        except Exception as e:
            print(f"[X] {e}")
    
    return questions


def generate_questions_per_document(
    documents_by_file: Dict[str, List[LangchainDocument]],
    questions_per_doc: int = 5,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    chunk_size: int = 512
) -> List[Dict]:
    """
    为每个文档生成指定数量的题目
    
    参数：
        documents_by_file: 按文件名分组的文档
        questions_per_doc: 每个文档的题目数量
        api_key: OpenAI API 密钥
        model: 模型名称
        chunk_size: 块大小（token 数）
    
    返回：
        所有生成的题目列表
        
    工作流程：
        1. 遍历每个文档
        2. 将文档分割成块
        3. 随机选择指定数量的块
        4. 为每个选中的块生成一道题
    """
    if api_key is None:
        api_key = get_api_key()
    
    if not api_key:
        print("[ERROR] No API key provided!")
        return []
    
    client = OpenAI(api_key=api_key)
    
    all_questions = []
    total_docs = len(documents_by_file)
    
    print(f"\nGenerating {questions_per_doc} questions for each of {total_docs} documents...")
    print(f"Expected total: {total_docs * questions_per_doc} questions")
    print("=" * 60)
    
    start_time = time.time()
    
    for idx, (doc_name, documents) in enumerate(documents_by_file.items(), 1):
        print(f"\n[{idx}/{total_docs}] {doc_name}")
        
        doc_questions = generate_questions_for_document(
            document_name=doc_name,
            documents=documents,
            num_questions=questions_per_doc,
            client=client,
            model=model,
            chunk_size=chunk_size
        )
        
        all_questions.extend(doc_questions)
        print(f"  [OK] Generated {len(doc_questions)} questions")
    
    print("\n" + "=" * 60)
    print(f"Total time: {format_time(time.time() - start_time)}")
    print(f"Successfully generated: {len(all_questions)} questions")
    
    return all_questions


# ============================================================================
# 保存函数
# ============================================================================

def save_questions_txt(questions: List[Dict], output_file: str = "questions.txt"):
    """将题目保存到文本文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTIPLE CHOICE QUESTIONS\n")
        f.write(f"Total: {len(questions)}\n")
        f.write("=" * 70 + "\n\n")
        
        for i, q in enumerate(questions, 1):
            f.write(f"Question {i}:\n{q.get('question', 'N/A')}\n\n")
            
            options = q.get('options', {})
            for letter in ['A', 'B', 'C', 'D']:
                if letter in options:
                    f.write(f"  {letter}. {options[letter]}\n")
            
            if 'correct_answer' in q:
                f.write(f"\nCorrect Answer: {q['correct_answer']}\n")
            
            # Handle both old format (explanation) and new format (explanations)
            if 'explanations' in q:
                f.write("\nExplanations:\n")
                explanations = q.get('explanations', {})
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in explanations:
                        f.write(f"  {letter}: {explanations[letter]}\n")
            elif 'explanation' in q:
                f.write(f"\nExplanation: {q['explanation']}\n")
            
            if 'source' in q:
                f.write(f"\nSource: {q['source']}\n")
            
            f.write("\n" + "-" * 70 + "\n\n")
    
    print(f"[OK] Saved: {output_file}")


def save_questions_json(questions: List[Dict], output_file: str = "questions.json"):
    """将题目保存到 JSON 文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数 - 为每个 PDF 文档生成题目
    
    配置参数：
        - pdf_directory: 包含 PDF 文件的目录
        - questions_per_doc: 每个文档的题目数量
        - model: OpenAI 模型
        - chunk_size: 文档块大小（token 数）
    """
    
    # ========================================================================
    # 配置参数
    # ========================================================================
    
    pdf_directory = "Clinical Guidelines"   # PDF 文件目录
    questions_per_doc = 5                   # 每个文档的题目数量
    model = "gpt-4o-mini"                   # OpenAI 模型
    chunk_size = 512                        # 每块的 token 数
    output_txt = "questions.txt"
    output_json = "questions.json"
    
    # ========================================================================
    # 开始
    # ========================================================================
    
    print("=" * 60)
    print("Medical Question Generator")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Directory: {pdf_directory}")
    print(f"Questions per document: {questions_per_doc}")
    print(f"Chunk size: {chunk_size} tokens")
    print("=" * 60)
    
    # 步骤 1：加载 PDF（按文档分组）
    print("\n[步骤 1] 正在加载 PDF 文件...")
    documents_by_file = load_pdfs_grouped(pdf_directory)
    
    if not documents_by_file:
        print("\n[ERROR] No documents loaded")
        return
    
    # 步骤 2：获取 API 密钥
    print("\n[步骤 2] 设置 API...")
    api_key = get_api_key()
    
    if not api_key:
        print("[ERROR] No API key provided")
        return
    
    # 步骤 3：为每个文档生成题目
    print("\n[步骤 3] 正在生成题目...")
    questions = generate_questions_per_document(
        documents_by_file,
        questions_per_doc=questions_per_doc,
        api_key=api_key,
        model=model,
        chunk_size=chunk_size
    )
    
    if not questions:
        print("\n[ERROR] No questions generated")
        return
    
    print(f"\n[OK] Successfully generated {len(questions)} questions!")
    
    # 步骤 4：保存
    print("\n[步骤 4] 保存结果...")
    save_questions_txt(questions, output_txt)
    save_questions_json(questions, output_json)
    
    print("\n" + "=" * 60)
    print("[DONE] Complete!")
    print(f"Processed {len(documents_by_file)} documents")
    print(f"Generated {len(questions)} questions")
    print("=" * 60)


if __name__ == "__main__":
    main()
