"""
使用 PyMuPDF + OpenAI Chat API 解析 Amboss PDF 文件

功能：
    1. 使用 PyMuPDF 从 PDF 提取文本
    2. 使用 OpenAI Chat API 将文本解析为结构化信息
    专门针对 Amboss 格式优化，只处理 Amboss 文件

优势：
    - 成本更低：文本 API 比 Vision API 便宜很多（约 1/10 成本）
    - 速度更快：文本处理比图片处理快
    - 更准确：直接从文本提取，避免 OCR 错误
    - 更智能：AI 能理解 PDF 的结构和内容
    - 处理复杂格式：能正确提取 Tip、选项、解释
    - 可以处理更多内容：文本没有图片大小限制

缺点：
    - 需要 API 调用费用（但比 Vision API 便宜很多）
    - 速度：API 调用比本地解析慢

模型说明：
    - gpt-5（默认）：最强模型，如果不可用会自动回退到 gpt-4o-2024-11-20
    - gpt-4o-2024-11-20：目前最强可用模型，最新版本，准确度最高
    - gpt-4o：GPT-4o 模型（自动使用最新版本）
    - gpt-4o-mini：轻量版，成本低（约 1/10），速度更快，准确度略低

使用方法：
    # 使用默认最强模型（GPT-5，如果不可用自动回退到 GPT-4o-2024-11-20）
    python parse_qbank_openai.py --dir "Qbanks and Practice Exams"
    
    # 使用轻量模型（降低成本）
    python parse_qbank_openai.py --dir "Qbanks and Practice Exams" --model gpt-4o-mini
    
    # 多页批量处理（提高效率）
    python parse_qbank_openai.py --dir "Qbanks and Practice Exams" --batch-size 3 --workers 5
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import concurrent.futures
import time

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
    
    # 尝试从 api-key.js 读取（项目根目录）
    api_key_path = Path(__file__).resolve().parent.parent / 'api-key.js'
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


def parse_pdf_with_openai(pdf_path: str, api_key: Optional[str] = None, model: str = "gpt-5",
                         batch_size: int = 0, max_workers: int = 3, overlap_pages: int = 2) -> List[Dict]:
    """
    使用 PyMuPDF 提取 PDF 文本，然后用 OpenAI Chat API 解析为结构化信息。
    
    优势：
    - 成本更低：文本 API 比 Vision API 便宜很多（约 1/10 成本）
    - 速度更快：文本处理比图片处理快
    - 更准确：直接从文本提取，避免 OCR 错误
    - 可以处理更多内容：文本没有图片大小限制
    
    Args:
        pdf_path: PDF 文件路径
        api_key: OpenAI API 密钥（如果为 None，尝试从环境变量或 api-key.js 读取）
        model: 使用的 OpenAI 模型（默认：gpt-5，如果不可用会自动回退到 gpt-4o-2024-11-20）
        batch_size: 每批处理的页数（0=一次性处理整个文件，1=逐页，2+=批量处理，默认：0）
        max_workers: 并发处理的线程数（默认：3，设置为 1 禁用并发。当 batch_size=0 时此参数无效）
        overlap_pages: 批次之间的重叠页数（默认：2，确保跨页题目不丢失）
        overlap_pages: 批次之间的重叠页数（默认：2，确保跨页题目不丢失）
    
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
    
    print(f"  Processing {len(doc)} pages with PyMuPDF + OpenAI Chat API...")
    print(f"  Requested model: {model}")
    print(f"  Batch size: {batch_size}, Max workers: {max_workers}")
    print(f"  Method: Text extraction (PyMuPDF) → Structured parsing (ChatGPT)")
    print()
    
    # 提取所有页面的文本并保存（连续文本，不按页分隔）
    # 同时提取图片并保存
    all_text_content = []
    page_images = {}  # 存储每页的图片路径 {page_num: [image_paths]}
    
    # 创建图片保存目录
    pdf_name = Path(pdf_path).stem
    images_dir = Path("images") / pdf_name
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        # 只添加文本内容，不添加页面分隔符，让文本连续
        if text.strip():  # 只添加非空页面
            all_text_content.append(text)
        
        # 提取当前页面的图片
        page_image_paths = []
        try:
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # 保存图片
                    image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    image_path = images_dir / image_filename
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # 使用相对路径（从项目根目录开始）
                    relative_path = f"images/{pdf_name}/{image_filename}"
                    page_image_paths.append(relative_path)
                except Exception as e:
                    print(f"    ⚠ Failed to extract image {img_index + 1} from page {page_num + 1}: {e}")
        
        except Exception as e:
            print(f"    ⚠ Error extracting images from page {page_num + 1}: {e}")
        
        # 保存该页的图片路径
        if page_image_paths:
            page_images[page_num + 1] = page_image_paths  # 页码从1开始
    
    # 保存提取的文本到文件（连续文本格式）
    pdf_name = Path(pdf_path).stem
    txt_output_path = f"{pdf_name}_extracted_text.txt"
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        # 用双换行符连接各页，保持段落结构
        f.write('\n\n'.join(all_text_content))
    print(f"  ✓ Extracted text saved to: {txt_output_path} (continuous text, not page-separated)")
    
    # 显示图片提取统计
    total_images = sum(len(imgs) for imgs in page_images.values())
    if total_images > 0:
        pages_with_images = len([p for p, imgs in page_images.items() if imgs])
        print(f"  ✓ Extracted {total_images} image(s) from {pages_with_images} page(s)")
    print()
    
    questions = []
    
    # 如果 batch_size 为 0 或负数，一次性处理整个文件
    if batch_size <= 0:
        # 一次性处理整个文件：读取已保存的 txt 文件
        print(f"  Processing entire document as a single batch...")
        print(f"  Reading extracted text from: {txt_output_path}")
        
        try:
            with open(txt_output_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # 一次性处理整个文档
            all_questions = process_full_document(
                client, full_text, pdf_path, model, len(doc), page_images
            )
            if all_questions:
                questions.extend(all_questions)
                print(f"  ✓ Successfully extracted {len(all_questions)} questions from full document")
            else:
                print(f"  ⚠ Warning: process_full_document returned empty list")
        except Exception as e:
            print(f"  ✗ Error processing full document: {e}")
            import traceback
            traceback.print_exc()
            return []
    else:
        # 分批处理：使用已提取的文本文件，按页分割后分批发送
        print(f"  Processing document in batches of {batch_size} pages...")
        print(f"  Reading extracted text from: {txt_output_path}")
        
        # 读取已提取的文本
        with open(txt_output_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # 将文本按页分割（使用已保存的页面文本）
        page_texts = all_text_content  # 已经提取的每页文本
        
        # 准备页面批次（带重叠）
        page_batches = []
        i = 0
        while i < len(page_texts):
            # 计算当前批次的结束位置
            end_idx = min(i + batch_size, len(page_texts))
            batch_texts = page_texts[i:end_idx]
            batch_page_nums = list(range(i + 1, end_idx + 1))
            
            # 添加重叠页面（除了第一页和最后一页）
            if i > 0 and overlap_pages > 0:
                # 添加前一个批次的最后几页作为重叠
                overlap_start = max(0, i - overlap_pages)
                overlap_texts = page_texts[overlap_start:i]
                overlap_nums = list(range(overlap_start + 1, i + 1))
                batch_texts = overlap_texts + batch_texts
                batch_page_nums = overlap_nums + batch_page_nums
            
            if end_idx < len(page_texts) and overlap_pages > 0:
                # 添加下一批次的开始几页作为重叠
                overlap_end = min(len(page_texts), end_idx + overlap_pages)
                overlap_texts = page_texts[end_idx:overlap_end]
                overlap_nums = list(range(end_idx + 1, overlap_end + 1))
                batch_texts = batch_texts + overlap_texts
                batch_page_nums = batch_page_nums + overlap_nums
            
            page_batches.append((batch_page_nums[0], batch_texts, batch_page_nums))
            
            # 移动到下一批次（考虑重叠）
            i += batch_size
        
        print(f"  Created {len(page_batches)} batches from {len(page_texts)} pages (overlap: {overlap_pages} pages)")
        
        # 处理文本批次（使用已提取的文本，而不是重新从 PDF 提取）
        if max_workers == 1:
            # 串行处理
            for batch_start_page, batch_texts, batch_page_nums in page_batches:
                batch_questions = process_text_batch(
                    client, batch_texts, batch_page_nums, batch_start_page, len(page_batches), pdf_path, model, page_images
                )
                questions.extend(batch_questions)
        else:
            # 并发处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_text_batch,
                        client, batch_texts, batch_page_nums, batch_start_page, len(page_batches), pdf_path, model, page_images
                    ): batch_start_page
                    for batch_start_page, batch_texts, batch_page_nums in page_batches
                }
                
                for future in concurrent.futures.as_completed(futures):
                    batch_num = futures[future]
                    try:
                        batch_questions = future.result()
                        questions.extend(batch_questions)
                    except Exception as e:
                        print(f"    ✗ Batch {batch_num} failed: {e}")
    
    doc.close()
    
    # 处理完成（模型信息已在每个批次处理时显示，不写入 JSON）
    if questions:
        print(f"\n  ✓ 处理完成，共提取 {len(questions)} 个题目")
    
    return questions


def process_full_document(client, full_text: str, pdf_path: str, model: str, total_pages: int, page_images: Dict[int, List[str]] = None) -> List[Dict]:
    """
    一次性处理整个文档的文本（从已保存的 txt 文件读取）。
    
    Args:
        client: OpenAI 客户端
        full_text: 完整的文档文本
        pdf_path: PDF 文件路径
        model: OpenAI 模型
        total_pages: 总页数
    
    Returns:
        List[Dict]: 提取的题目列表
    """
    print(f"    Processing entire document ({total_pages} pages)...", end=" ")
    
    system_prompt = """You are a professional medical exam question parsing expert. Please carefully analyze the text extracted from the PDF and accurately extract all question information.

**IMPORTANT: All output must be in English. All explanations, tips, and text content must be written in English only.**

For each question, please extract the following information:
1. question: Question text (including complete stem, all lab data, Tip. Tip should be included in the question stem)
2. options: Options dictionary, format {"A": "Option A text", "B": "Option B text", ...} (supports A-J, up to 10 options)
   - **Must extract all options**, do not miss any option
   - Option text should be complete, including all content after the option label
   - Ignore percentages after options (e.g., "69%", "6%", etc.)
3. correct_answer: Correct answer letter (A-J), usually the first option (because the first option is usually the correct answer)
4. tip: Tip text (if Tip is already included in the question stem, this can be an empty string "")
5. explanations: Explanation object (**CRITICAL: MUST extract EXACTLY from original text, word-for-word**)
   - Format: {"A": "Explanation for option A", "B": "Explanation for option B", "C": "Explanation for option C", ...}
   - **CRITICAL RULE: All explanations MUST be copied EXACTLY as they appear in the original PDF text. DO NOT generate, rewrite, summarize, or paraphrase explanations. Copy them word-for-word from the source text.**
   - **If an explanation for an option does not exist in the original text, use an empty string "". DO NOT create your own explanation.**
   - **Must extract explanations for all options that exist in the original text**
   - **Each explanation must match the original text exactly, character-by-character if possible**
   - **All explanations must be in English (if the original text is in English)**
6. page_number: Page number where the question is located (estimated based on the question's position in the document, starting from 1)

Output format: Must be a JSON object containing a "questions" field with an array of questions.

**Key Requirements (Very Important):**
1. **Question MUST start with a question number**: Every question text MUST begin with a number followed by a period and space (e.g., "1. ", "2. ", "25. "). If a question does not have a number at the start, DO NOT extract it as a separate question. Only extract questions that clearly start with a number.
2. **Must extract all options**: If a question has 5 options (A-E), all must be extracted, none can be missed
3. **Explanations MUST be extracted EXACTLY from original text**: 
   - **DO NOT generate, rewrite, summarize, or paraphrase explanations**
   - **Copy explanations word-for-word from the source text**
   - **If an explanation does not exist in the original text, use empty string ""**
   - **DO NOT create your own explanations**
4. **Explanations must be complete**: Explanation text must be complete, not truncated, including explanations for all options that exist in the original text
5. **Question must be complete**: If a question spans multiple pages, all parts (stem, options, explanations) must be completely extracted. The question text must include the complete stem starting with the question number.
6. **Lab data must be included**: All laboratory test results, vital signs, etc. must be included in the question field
7. **Option format**: Option text should only contain option content, not percentages, explanations, etc.
8. **Do NOT extract incomplete questions**: If a question fragment does not start with a number (e.g., only options, only explanations, or partial text), DO NOT extract it as a separate question. Only extract complete questions that start with a number.
9. **Page number estimation**: Estimate a reasonable page number based on the question's position in the document (starting from 1)
10. **All text must be in English**: All explanations, tips, and content must be written in English only

Please carefully check each question to ensure it is complete and no information is missing. Extract all questions from the document."""
    
    user_prompt = f"""Please analyze the following complete text extracted from the PDF and extract all question information. Strictly follow the required JSON format for output.

**IMPORTANT: All explanations, tips, and text content must be in English only.**

**CRITICAL: For explanations field:**
- **MUST copy explanations EXACTLY as they appear in the original text, word-for-word**
- **DO NOT generate, rewrite, summarize, or paraphrase explanations**
- **If an explanation for an option does not exist in the original text, use empty string ""**
- **DO NOT create your own explanations**

完整文本内容：
{full_text}"""
    
    # 构建消息内容：纯文本
    user_content = user_prompt
    
    # 调用 OpenAI Chat API
    actual_model = model
    
    # GPT-5 和 o1 系列不支持 temperature 参数，只能使用默认值
    # GPT-5 使用 max_completion_tokens 而不是 max_tokens
    api_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    # 根据模型类型设置不同的参数
    if isinstance(model, str) and model.startswith(("gpt-5", "o1")):
        # GPT-5 和 o1 系列使用 max_completion_tokens
        api_params["max_completion_tokens"] = 16000  # 增加 token 限制，因为是一次性处理整个文档
    else:
        # 其他模型使用 max_tokens 和 temperature
        api_params["max_tokens"] = 16000
        api_params["temperature"] = 0.1  # 低温度以确保准确性
    
    try:
        response = client.chat.completions.create(**api_params)
        if hasattr(response, 'model'):
            actual_model = response.model
    except Exception as e:
        error_str = str(e).lower()
        error_message = str(e)
        
        # 如果是 gpt-5 的参数错误，尝试使用正确的参数重试
        if model == "gpt-5" and ("temperature" in error_str or "max_tokens" in error_str):
            if "max_tokens" in error_str:
                print(f"\n  ℹ GPT-5 不支持 max_tokens，使用 max_completion_tokens 重试...\n")
            else:
                print(f"\n  ℹ GPT-5 不支持 temperature 参数，使用默认值重试...\n")
            # 重试时使用 GPT-5 支持的参数
            api_params_fixed = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 16000
            }
            try:
                response = client.chat.completions.create(**api_params_fixed)
                if hasattr(response, 'model'):
                    actual_model = response.model
            except Exception as e2:
                print(f"\n  ✗ GPT-5 API 调用失败: {e2}")
                print(f"  详细错误信息: {error_message}")
                # 检查是否是模型不存在
                if any(keyword in error_str for keyword in ["not found", "invalid", "does not exist", "model"]):
                    print(f"\n  ⚠ 提示: gpt-5 模型可能不存在或不可用")
                    print(f"  建议: 尝试使用 'gpt-4o-2024-11-20' 或其他可用模型")
                raise e2
        # 检查是否是 gpt-5 模型不存在
        elif model == "gpt-5" and any(keyword in error_str for keyword in ["not found", "invalid", "does not exist", "model"]):
            print(f"\n  ✗ GPT-5 模型不可用: {error_message}")
            print(f"  ⚠ 提示: gpt-5 模型可能不存在或不可用")
            print(f"  建议: 尝试使用 'gpt-4o-2024-11-20' 或其他可用模型")
            raise
        else:
            print(f"\n  ✗ API 调用失败: {error_message}")
            raise
    
    # 解析响应
    if not response.choices or not response.choices[0].message:
        print(f"✗ Error: API returned empty response")
        print(f"    Response object type: {type(response)}")
        print(f"    Response has choices: {hasattr(response, 'choices')}")
        if hasattr(response, 'choices'):
            print(f"    Choices length: {len(response.choices) if response.choices else 0}")
        return []
    
    response_text = response.choices[0].message.content
    
    # 检查响应是否为空
    if not response_text:
        print(f"✗ Error: API returned empty content")
        print(f"    Response object: {response}")
        print(f"    Message object: {response.choices[0].message}")
        print(f"    Finish reason: {response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else 'N/A'}")
        return []
    
    # 检查响应长度
    if len(response_text.strip()) == 0:
        print(f"✗ Error: API returned empty string")
        print(f"    Finish reason: {response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else 'N/A'}")
        return []
    
    try:
        # 尝试解析 JSON
        result = json.loads(response_text)
        
        # 提取题目列表
        if isinstance(result, dict):
            if 'questions' in result:
                all_questions = result['questions']
            elif 'items' in result:
                all_questions = result['items']
            elif 'data' in result:
                all_questions = result['data']
            else:
                values = list(result.values())
                if len(values) == 1 and isinstance(values[0], list):
                    all_questions = values[0]
                else:
                    all_questions = []
        elif isinstance(result, list):
            all_questions = result
        else:
            all_questions = []
        
        # 处理每个题目
        for q in all_questions:
            if isinstance(q, dict):
                # 添加元数据
                q['source'] = "PyMuPDF + OpenAI Chat API (Full Document)"
                q['source_file'] = Path(pdf_path).name
                
                # 如果没有 page_number，尝试从题目编号估算
                if 'page_number' not in q or not q.get('page_number'):
                    # 简单估算：根据题目编号（1-25）估算页码
                    question_num = None
                    if 'question' in q:
                        import re
                        match = re.search(r'^(\d+)\.', q['question'])
                        if match:
                            question_num = int(match.group(1))
                            # 简单估算：每页约 1-2 题
                            q['page_number'] = min((question_num - 1) // 2 + 1, total_pages)
                    if 'page_number' not in q:
                        q['page_number'] = 1
                
                # 添加图片路径（根据页码匹配）
                page_num = q.get('page_number')
                if page_images and page_num in page_images:
                    q['images'] = page_images[page_num]
                else:
                    q['images'] = []  # 如果没有图片，添加空数组
                
                # 确保字段存在
                if 'question' not in q:
                    continue
                if 'options' not in q:
                    q['options'] = {}
                if 'correct_answer' not in q:
                    q['correct_answer'] = 'A'
                if 'tip' not in q:
                    q['tip'] = ""
                if 'explanations' not in q:
                    q['explanations'] = {}
                # 如果存在旧的 explanation 字段，转换为 explanations 格式
                if 'explanation' in q and q['explanation'] and not q.get('explanations'):
                    q['explanations'] = parse_explanation_to_dict(q['explanation'], q.get('options', {}), q.get('correct_answer', 'A'))
                    del q['explanation']
        
        # 显示实际使用的模型
        model_info = ""
        if actual_model != model:
            model_info = f" [回退到: {actual_model}]"
        elif actual_model:
            model_info = f" [使用: {actual_model}]"
        
        print(f"✓ Extracted {len(all_questions)} question(s){model_info}")
        
        # 调试信息：如果题目数量为 0，显示更多信息
        if len(all_questions) == 0:
            print(f"  ⚠ Warning: No questions extracted. Response preview:")
            print(f"     {response_text[:500]}...")
        
        return all_questions
        
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse JSON: {e}")
        print(f"    Response length: {len(response_text)} characters")
        print(f"    Response preview (first 500 chars):")
        print(f"    {response_text[:500]}")
        if len(response_text) > 500:
            print(f"    ... (truncated, showing last 200 chars)")
            print(f"    {response_text[-200:]}")
        # 尝试保存完整响应到文件以便调试
        try:
            debug_file = "api_response_debug.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"Full API Response:\n")
                f.write(f"{'='*80}\n")
                f.write(response_text)
            print(f"    Full response saved to: {debug_file}")
        except:
            pass
        return []
    except Exception as e:
        print(f"✗ Error: {e}")
        return []


def process_text_batch(client, batch_texts: List[str], batch_page_nums: List[int], 
                       batch_num: int, total_batches: int, pdf_path: str, model: str, page_images: Dict[int, List[str]] = None) -> List[Dict]:
    """
    处理一批文本：使用已提取的文本，然后用 ChatGPT 解析为结构化信息。
    
    这是 process_page_batch 的优化版本，直接使用已提取的文本，避免重复提取。
    
    Args:
        client: OpenAI 客户端
        batch_texts: 文本列表（每页的文本）
        batch_page_nums: 对应的页码列表
        batch_num: 批次编号
        total_batches: 总批次数
        pdf_path: PDF 文件路径
        model: OpenAI 模型
    
    Returns:
        List[Dict]: 提取的题目列表
    """
    try:
        # 构建提示词
        if len(batch_texts) == 1:
            page_info = f"第 {batch_page_nums[0]} 页"
            print(f"    Processing page {batch_page_nums[0]}...", end=" ")
        else:
            page_info = f"第 {batch_page_nums[0]}-{batch_page_nums[-1]} 页（共 {len(batch_texts)} 页）"
            print(f"    Processing pages {batch_page_nums[0]}-{batch_page_nums[-1]} ({len(batch_texts)} pages)...", end=" ")
        
        # 合并所有页面的文本
        combined_text = "\n\n".join([f"[Page {batch_page_nums[i]}]\n{text}" for i, text in enumerate(batch_texts)])
        
        system_prompt = """You are a professional medical exam question parsing expert. Please carefully analyze the text extracted from the PDF and accurately extract all question information.

**IMPORTANT: All output must be in English. All explanations, tips, and text content must be written in English only.**

**Output Format Requirements (Strictly Follow):**
Must be a JSON object, format as follows:
{
  "questions": [
    {
      "question": "Complete question text, including all lab data, vital signs, Tip, etc.",
      "options": {"A": "Complete option A text", "B": "Complete option B text", "C": "Complete option C text", ...},
      "correct_answer": "A",
      "tip": "Tip text (if Tip is already in the question stem, can be empty string)",
      "explanations": {
        "A": "Explanation for option A (why correct or incorrect)",
        "B": "Explanation for option B (why correct or incorrect)",
        "C": "Explanation for option C (why correct or incorrect)",
        ...
      },
      "page_number": 1
    }
  ]
}

**Each question must contain the following fields (all fields are required):**
1. question: Question text (**must be complete**, including complete stem, all lab data, vital signs, Tip, etc.)
2. options: Options dictionary (**must include all options**, format {"A": "Option A text", "B": "Option B text", ...}, supports A-J)
   - **Absolutely cannot miss any option**, if a question has 5 options (A-E), all must be extracted
   - Option text must be complete, including all content after the option label
   - Ignore percentages after options (e.g., "69%", "6%", etc.)
3. correct_answer: Correct answer letter (A-J, **must provide**)
4. tip: Tip text (if Tip is already included in the question stem, this can be an empty string "")
5. explanations: Explanation object (**CRITICAL: MUST extract EXACTLY from original text, word-for-word**)
   - Format: {"A": "Explanation for option A", "B": "Explanation for option B", "C": "Explanation for option C", ...}
   - **CRITICAL RULE: All explanations MUST be copied EXACTLY as they appear in the original PDF text. DO NOT generate, rewrite, summarize, or paraphrase explanations. Copy them word-for-word from the source text.**
   - **If an explanation for an option does not exist in the original text, use an empty string "". DO NOT create your own explanation.**
   - **Must extract explanations for all options that exist in the original text**
   - **Each explanation must match the original text exactly, character-by-character if possible**
   - **All explanations must be in English (if the original text is in English)**
6. page_number: Page number where the question is located (determined from the provided page range)

**Key Requirements (Very Important, violation will cause extraction failure):**
1. **Question MUST start with a question number**: Every question text MUST begin with a number followed by a period and space (e.g., "1. ", "2. ", "25. "). If a question does not have a number at the start, DO NOT extract it as a separate question. Only extract questions that clearly start with a number.
2. **All fields are required**: question, options, correct_answer, explanations cannot be empty
3. **Must extract all options**: If a question has 5 options (A-E), all must be extracted, none can be missed
4. **Explanations MUST be extracted EXACTLY from original text**: 
   - **DO NOT generate, rewrite, summarize, or paraphrase explanations**
   - **Copy explanations word-for-word from the source text**
   - **If an explanation does not exist in the original text, use empty string ""**
   - **DO NOT create your own explanations**
5. **Question must be complete**: If a question spans multiple pages, all parts (stem, options, explanations) must be completely extracted. The question text must include the complete stem starting with the question number.
6. **Lab data must be included**: All laboratory test results, vital signs, etc. must be included in the question field
7. **Option format**: Option text should only contain option content, not percentages, explanations, etc.
8. **Do NOT extract incomplete questions**: If a question fragment does not start with a number (e.g., only options, only explanations, or partial text), DO NOT extract it as a separate question. Only extract complete questions that start with a number.
9. **All text must be in English**: All explanations, tips, and content must be written in English only

**Important Reminders:**
- If complete question information is not found in the text (e.g., missing options or explanations), try to find from adjacent pages
- If complete information cannot be found, still extract available information, but ensure all fields exist (use empty string or empty object for missing fields)
- Do not skip questions because they are incomplete, extract all available information as much as possible

Please carefully check each question to ensure it is complete and no information is missing. Extract all questions from the document."""
        
        user_prompt = f"""Please analyze the following text extracted from the PDF ({page_info}) and extract all question information. Strictly follow the required JSON format for output.

**IMPORTANT: All explanations, tips, and text content must be in English only.**

**CRITICAL: For explanations field:**
- **MUST copy explanations EXACTLY as they appear in the original text, word-for-word**
- **DO NOT generate, rewrite, summarize, or paraphrase explanations**
- **If an explanation for an option does not exist in the original text, use empty string ""**
- **DO NOT create your own explanations**

文本内容：
{combined_text}"""
        
        # 调用 OpenAI Chat API
        actual_model = model
        
        # GPT-5 和 o1 系列不支持 temperature 参数，只能使用默认值
        # GPT-5 使用 max_completion_tokens 而不是 max_tokens
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        # 根据模型类型设置不同的参数
        if isinstance(model, str) and model.startswith(("gpt-5", "o1")):
            # GPT-5 和 o1 系列使用 max_completion_tokens
            api_params["max_completion_tokens"] = 16000
        else:
            # 其他模型使用 max_tokens 和 temperature
            api_params["max_tokens"] = 16000
            api_params["temperature"] = 0.1  # 低温度以确保准确性
        
        try:
            response = client.chat.completions.create(**api_params)
            if hasattr(response, 'model'):
                actual_model = response.model
        except Exception as e:
            error_str = str(e).lower()
            error_message = str(e)
            
            # 如果是 gpt-5 的参数错误，尝试使用正确的参数重试
            if model == "gpt-5" and ("temperature" in error_str or "max_tokens" in error_str):
                if "max_tokens" in error_str:
                    print(f"\n  ℹ GPT-5 不支持 max_tokens，使用 max_completion_tokens 重试...\n")
                else:
                    print(f"\n  ℹ GPT-5 不支持 temperature 参数，使用默认值重试...\n")
                # 重试时使用 GPT-5 支持的参数
                api_params_fixed = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": 16000
                }
                response = client.chat.completions.create(**api_params_fixed)
                if hasattr(response, 'model'):
                    actual_model = response.model
            else:
                raise
        
        # 解析响应
        if not response.choices or not response.choices[0].message:
            print(f"✗ Error: API returned empty response")
            return []
        
        response_text = response.choices[0].message.content
        
        if not response_text:
            print(f"✗ Error: API returned empty content")
            return []
        
        try:
            # 尝试解析 JSON
            result = json.loads(response_text)
            
            # 提取题目列表
            if isinstance(result, dict):
                if 'questions' in result:
                    batch_questions = result['questions']
                elif 'items' in result:
                    batch_questions = result['items']
                elif 'data' in result:
                    batch_questions = result['data']
                else:
                    values = list(result.values())
                    if len(values) == 1 and isinstance(values[0], list):
                        batch_questions = values[0]
                    else:
                        batch_questions = []
            elif isinstance(result, list):
                batch_questions = result
            else:
                batch_questions = []
            
            # 处理每个题目
            valid_questions = []
            for q in batch_questions:
                if isinstance(q, dict):
                    # 验证题目是否有题号
                    question_text = q.get('question', '').strip()
                    if not question_text:
                        continue
                    
                    # 检查题目是否以题号开头
                    import re
                    if not re.match(r'^\d+\.\s+', question_text):
                        # 跳过没有题号的题目（可能是片段或不完整提取）
                        continue
                    
                    # 添加元数据
                    q['source'] = "PyMuPDF + OpenAI Chat API"
                    q['source_file'] = Path(pdf_path).name
                    
                    # 如果没有 page_number，使用批次中的页码
                    if 'page_number' not in q or not q.get('page_number'):
                        q['page_number'] = batch_page_nums[0] if batch_page_nums else 1
                    
                    # 添加图片路径（根据页码匹配）
                    page_num = q.get('page_number')
                    if page_images and page_num in page_images:
                        q['images'] = page_images[page_num]
                    else:
                        q['images'] = []  # 如果没有图片，添加空数组
                    
                    # 确保字段存在
                    if 'options' not in q:
                        q['options'] = {}
                    if 'correct_answer' not in q:
                        q['correct_answer'] = 'A'
                    if 'tip' not in q:
                        q['tip'] = ""
                    # explanation 字段：从 explanations 中提取正确答案的解释，如果没有则留空
                    if 'explanation' not in q or not q.get('explanation'):
                        correct_answer = q.get('correct_answer', 'A')
                        explanations = q.get('explanations', {})
                        if correct_answer in explanations:
                            q['explanation'] = explanations[correct_answer]
                        else:
                            q['explanation'] = ""
                    
                    valid_questions.append(q)
            
            # 使用验证后的题目列表
            batch_questions = valid_questions
            
            # 显示实际使用的模型
            model_info = ""
            if actual_model != model:
                model_info = f" [回退到: {actual_model}]"
            elif actual_model:
                model_info = f" [使用: {actual_model}]"
            
            print(f"✓ Extracted {len(batch_questions)} question(s){model_info}")
            
            return batch_questions
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON: {e}")
            print(f"    Response preview: {response_text[:200]}...")
            return []
        except Exception as e:
            print(f"✗ Error: {e}")
            return []
            
    except Exception as e:
        print(f"✗ Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_page_batch(client, batch_pages: List, batch_num: int, total_batches: int,
                      pdf_path: str, model: str) -> List[Dict]:
    """
    处理一批页面：使用 PyMuPDF 提取文本，然后用 ChatGPT 解析为结构化信息。
    
    优势：
    1. 成本更低：文本 API 比 Vision API 便宜很多
    2. 速度更快：文本处理比图片处理快
    3. 可以处理更多内容：文本没有图片大小限制
    4. 更准确：直接从文本提取，避免 OCR 错误
    
    Args:
        client: OpenAI 客户端
        batch_pages: 页面列表（fitz.Page 对象）
        batch_num: 批次编号
        total_batches: 总批次数
        pdf_path: PDF 文件路径
        model: OpenAI 模型
    
    Returns:
        List[Dict]: 提取的题目列表
    """
    try:
        import fitz  # PyMuPDF
        
        # 提取文本（每页）
        page_texts = []
        page_nums = []
        
        for i, page in enumerate(batch_pages):
            # 使用 PyMuPDF 提取文本
            text = page.get_text()
            page_texts.append(text)
            page_nums.append(batch_num + i)
        
        # 构建提示词
        if len(batch_pages) == 1:
            page_info = f"第 {page_nums[0]} 页"
            print(f"    Processing page {page_nums[0]}/{total_batches}...", end=" ")
        else:
            page_info = f"第 {page_nums[0]}-{page_nums[-1]} 页（共 {len(batch_pages)} 页）"
            print(f"    Processing pages {page_nums[0]}-{page_nums[-1]} ({len(batch_pages)} pages)...", end=" ")
        
        # 合并所有页面的文本
        combined_text = "\n\n".join([f"[Page {page_nums[i]}]\n{text}" for i, text in enumerate(page_texts)])
        
        system_prompt = """You are a professional medical exam question parsing expert. Please carefully analyze the text extracted from the PDF and accurately extract all question information.

**IMPORTANT: All output must be in English. All explanations, tips, and text content must be written in English only.**

For each question, please extract the following information:
1. question: Question text (including complete stem, all lab data, Tip. Tip should be included in the question stem)
2. options: Options dictionary, format {"A": "Option A text", "B": "Option B text", ...} (supports A-J, up to 10 options)
   - **Must extract all options**, do not miss any option
   - Option text should be complete, including all content after the option label
   - Ignore percentages after options (e.g., "69%", "6%", etc.)
3. correct_answer: Correct answer letter (A-J), usually the first option (because the first option is usually the correct answer)
4. tip: Tip text (if Tip is already included in the question stem, this can be an empty string "")
5. explanations: Explanation object (**must provide a separate explanation for each option**)
   - Format: {"A": "Explanation for option A", "B": "Explanation for option B", "C": "Explanation for option C", ...}
   - **Must include explanations for all options**, clearly stating why each option is correct or incorrect
   - Each option's explanation should be detailed, explaining why it is correct or incorrect
   - Must include detailed explanation for the correct answer
   - If the text contains explanations for other options, they must also be included
   - **All explanations must be in English**
6. page_number: Page number where the question is located (determined from the provided page range)

Output format: Must be a JSON object containing a "questions" field with an array of questions.
Format example:
{
  "questions": [
    {
      "question": "Complete question text, including all lab data and Tip...",
      "options": {"A": "Complete option A text", "B": "Complete option B text", "C": "Complete option C text", "D": "Complete option D text", "E": "Complete option E text"},
      "correct_answer": "A",
      "tip": "",
      "explanations": {
        "A": "Detailed explanation why A is correct",
        "B": "Detailed explanation why B is incorrect",
        "C": "Detailed explanation why C is incorrect",
        ...
      },
      "page_number": 1
    }
  ]
}

**Key Requirements (Very Important):**
1. **Must extract all options**: If a question has 5 options (A-E), all must be extracted, none can be missed
2. **Explanations must correspond to each option**: **Must provide a separate explanation for each option**, clearly stating why the option is correct or incorrect
3. **Explanations must be complete**: Explanation text must be complete, not truncated, including explanations for all options
4. **Question must be complete**: If a question spans multiple pages, all parts (stem, options, explanations) must be completely extracted
5. **Lab data must be included**: All laboratory test results, vital signs, etc. must be included in the question field
6. **Option format**: Option text should only contain option content, not percentages, explanations, etc.
7. **If question is incomplete** (missing options or explanations), try to find from subsequent pages, or mark as incomplete
8. **All text must be in English**: All explanations, tips, and content must be written in English only

如果页面中有多个题目，请提取所有题目。
如果页面中没有题目，返回 {"questions": []}。

请仔细检查每个题目是否完整，确保没有遗漏任何信息。"""
        
        user_prompt = f"""请分析以下从 PDF 提取的文本（{page_info}），提取所有题目信息。严格按照要求的 JSON 格式输出。

文本内容：
{combined_text}"""
        
        # 构建消息内容：纯文本（不使用 Vision API）
        user_content = user_prompt
        
        # 调用 OpenAI Chat API（文本处理，不使用 Vision API）
        # 如果 gpt-5 不可用，自动回退到 gpt-4o
        actual_model = model  # 记录实际使用的模型
        
        # GPT-5 和 o1 系列不支持 temperature 参数，只能使用默认值
        # GPT-5 使用 max_completion_tokens 而不是 max_tokens
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        # 根据模型类型设置不同的参数
        if isinstance(model, str) and model.startswith(("gpt-5", "o1")):
            # GPT-5 和 o1 系列使用 max_completion_tokens
            api_params["max_completion_tokens"] = 16000
        else:
            # 其他模型使用 max_tokens 和 temperature
            api_params["max_tokens"] = 16000
            api_params["temperature"] = 0.1  # 低温度以确保准确性
        
        try:
            response = client.chat.completions.create(**api_params)
            # 从响应中获取实际使用的模型
            if hasattr(response, 'model'):
                actual_model = response.model
        except Exception as e:
            # 如果模型不可用，尝试回退到其他版本
            error_str = str(e).lower()
            error_message = str(e)
            
            # 如果是指定版本不可用，尝试回退到基础版本
            if isinstance(model, str) and model.startswith("gpt-4o-") and ("not found" in error_str or "invalid" in error_str or "does not exist" in error_str or "model" in error_str):
                if batch_num == 1:  # 只在第一批显示详细错误
                    print(f"\n  ⚠ {model} 不可用，尝试回退到 gpt-4o...")
                
                print(f"⚠ 回退到 gpt-4o...", end=" ")
                actual_model = "gpt-4o"
                model = "gpt-4o"
            # 如果是 gpt-5 的参数错误（如 temperature 或 max_tokens），尝试使用正确的参数重试
            elif model == "gpt-5" and ("temperature" in error_str or "max_tokens" in error_str):
                if batch_num == 1:  # 只在第一批显示
                    if "max_tokens" in error_str:
                        print(f"\n  ℹ GPT-5 不支持 max_tokens，使用 max_completion_tokens 重试...\n")
                    else:
                        print(f"\n  ℹ GPT-5 不支持 temperature 参数，使用默认值重试...\n")
                # 重试时使用 GPT-5 支持的参数
                api_params_fixed = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ],
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": 16000  # GPT-5 使用 max_completion_tokens
                }
                response = client.chat.completions.create(**api_params_fixed)
                if hasattr(response, 'model'):
                    actual_model = response.model
            # 如果是 gpt-5 不可用（模型不存在），回退到 gpt-4o-2024-11-20
            elif model == "gpt-5" and ("not found" in error_str or "invalid" in error_str or "does not exist" in error_str or "model" in error_str):
                if batch_num == 1:  # 只在第一批显示详细错误
                    print(f"\n  ⚠ GPT-5 不可用 - 详细错误信息:")
                    print(f"     {error_message}")
                    print(f"\n  ℹ 可能的原因:")
                    print(f"     1. 模型尚未发布到 API")
                    print(f"        → GPT-5 可能还未正式发布，将回退到 GPT-4o-2024-11-20")
                    print(f"     2. 模型名称不正确")
                    print(f"        → 可能需要带日期戳，如 'gpt-5-2025-08-07'（而非 'gpt-5'）")
                    print(f"        → 或者模型名称可能是 'o1', 'o3' 等其他名称")
                    print(f"     3. 账户权限限制")
                    print(f"        → 账户创建时间需超过 7 天")
                    print(f"        → 需要支付至少 $0.50 的预付费额度")
                    print(f"        → 可能需要企业账户验证")
                    print(f"     4. 区域限制")
                    print(f"        → 某些地区可能无法访问 GPT-5")
                    print(f"     5. API SDK 版本过旧")
                    print(f"        → 需要 OpenAI Python SDK 4.0.0+ 版本")
                    print(f"\n  → 自动回退到 GPT-4o-2024-11-20（目前最强的可用模型）\n")
                
                print(f"⚠ 回退到 gpt-4o-2024-11-20...", end=" ")
                actual_model = "gpt-4o-2024-11-20"
                model = "gpt-4o-2024-11-20"
            
            # 如果回退到其他模型，重试请求
            if model in ["gpt-4o", "gpt-4o-2024-11-20"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=16000  # 增加最大 token 数，确保解释不会被截断
                )
                # 从响应中获取实际使用的模型
                if hasattr(response, 'model'):
                    actual_model = response.model
            else:
                raise  # 重新抛出其他错误
        
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
                    q['source'] = "PyMuPDF + OpenAI Chat API"
                    q['source_file'] = Path(pdf_path).name
                    
                    # 如果 AI 没有提供 page_number，使用批次的第一页
                    if 'page_number' not in q or not q.get('page_number'):
                        q['page_number'] = page_nums[0]
                    else:
                        # 确保 page_number 在有效范围内
                        ai_page = int(q.get('page_number', page_nums[0]))
                        if ai_page not in page_nums:
                            # 如果 AI 提供的页码不在批次中，使用最接近的页码
                            q['page_number'] = min(page_nums, key=lambda x: abs(x - ai_page))
                    
                    # 确保字段存在
                    if 'question' not in q:
                        continue
                    if 'options' not in q:
                        q['options'] = {}
                    if 'correct_answer' not in q:
                        q['correct_answer'] = 'A'
                    if 'tip' not in q:
                        q['tip'] = ""
                    # explanation 字段：从 explanations 中提取正确答案的解释，如果没有则留空
                    if 'explanation' not in q or not q.get('explanation'):
                        correct_answer = q.get('correct_answer', 'A')
                        explanations = q.get('explanations', {})
                        if correct_answer in explanations:
                            q['explanation'] = explanations[correct_answer]
                        else:
                            q['explanation'] = ""
            
            # 显示实际使用的模型
            model_info = ""
            if actual_model != model:
                model_info = f" [回退到: {actual_model}]"
            elif actual_model:
                model_info = f" [使用: {actual_model}]"
            
            print(f"✓ Extracted {len(page_questions)} question(s){model_info}")
            
            # 不在 JSON 中添加模型信息，只在控制台显示
            return page_questions
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse JSON: {e}")
            print(f"    Response preview: {response_text[:200]}...")
            return []
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return []


def parse_explanation_to_dict(explanation_text: str, options: Dict[str, str], correct_answer: str) -> Dict[str, str]:
    """
    将字符串格式的解释转换为对象格式（每个选项对应一个解释）。
    
    Args:
        explanation_text: 解释文本（可能包含多个选项的解释）
        options: 选项字典
        correct_answer: 正确答案
    
    Returns:
        Dict[str, str]: 每个选项对应的解释
    """
    import re
    
    explanations = {}
    
    # 尝试从文本中提取每个选项的解释
    # 查找格式：**选项A：** 或 选项A： 或 A: 等
    for option_key in options.keys():
        # 尝试多种匹配模式
        patterns = [
            rf'\*\*选项{option_key}：?\*\*[：:]?\s*(.*?)(?=\*\*选项|\Z)',
            rf'选项{option_key}：?\s*(.*?)(?=选项[A-Z]：|\Z)',
            rf'{option_key}[：:]\s*(.*?)(?=[A-Z][：:]|\Z)',
        ]
        
        found = False
        for pattern in patterns:
            match = re.search(pattern, explanation_text, re.DOTALL | re.IGNORECASE)
            if match:
                explanations[option_key] = match.group(1).strip()
                found = True
                break
        
        # 如果没有找到，根据选项是否为正确答案生成默认解释
        if not found:
            if option_key == correct_answer:
                explanations[option_key] = explanation_text[:500] if explanation_text else "正确答案。"
            else:
                explanations[option_key] = "此选项不正确。"
    
    return explanations


def validate_question(q: Dict) -> bool:
    """
    验证题目是否有效。
    
    Args:
        q: 题目字典
    
    Returns:
        bool: 如果题目有效返回True，否则返回False
    """
    import re
    
    question_text = q.get('question', '').strip()
    if not question_text:
        return False
    
    # 必须包含题号
    if not re.match(r'^\d+\.\s+', question_text):
        return False
    
    # 必须有选项
    options = q.get('options', {})
    if not options or len(options) == 0:
        return False
    
    # 必须有正确答案
    correct_answer = q.get('correct_answer', '').strip()
    if not correct_answer:
        return False
    
    return True


def deduplicate_questions(questions: List[Dict]) -> List[Dict]:
    """
    去除重复的题目。
    
    判断重复的标准（优先级从高到低）：
    1. 题目编号相同（最高优先级）
    2. 题目文本相似（去除空格和标点后前100个字符相同）
    
    Args:
        questions: 题目列表
    
    Returns:
        List[Dict]: 去重后的题目列表
    """
    import re
    
    # 先过滤无效题目
    valid_questions = [q for q in questions if validate_question(q)]
    
    # 按题号分组
    questions_by_number = {}
    questions_without_number = []
    
    for q in valid_questions:
        question_text = q.get('question', '').strip()
        match = re.match(r'^(\d+)\.\s+', question_text)
        if match:
            question_num = match.group(1)
            if question_num not in questions_by_number:
                questions_by_number[question_num] = []
            questions_by_number[question_num].append(q)
        else:
            questions_without_number.append(q)
    
    # 对于每个题号，保留最完整的版本
    unique_questions = []
    for question_num, q_list in questions_by_number.items():
        if len(q_list) == 1:
            unique_questions.append(q_list[0])
        else:
            # 多个相同题号的题目，保留最完整的
            best_q = max(q_list, key=lambda q: (
                len(q.get('options', {})),
                len(q.get('explanations', {})),
                len(q.get('question', ''))
            ))
            unique_questions.append(best_q)
    
    # 对于没有题号的题目，基于文本相似度去重
    seen_texts = set()
    for q in questions_without_number:
        question_text = q.get('question', '').strip()
        normalized_text = re.sub(r'[^\w]', '', question_text[:100]).lower()
        if normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            unique_questions.append(q)
    
    return unique_questions


def parse_directory(directory: str, output_file: str = 'qbank_amboss_openai.json',
                   api_key: Optional[str] = None, model: str = "gpt-5", batch_size: int = 3, max_workers: int = 3, overlap_pages: int = 2, force: bool = False) -> List[Dict]:
    """
    批量解析目录下的所有 PDF 文件（完全使用 OpenAI）。
    
    Args:
        directory: PDF 文件目录路径
        output_file: 输出 JSON 文件路径
        api_key: OpenAI API 密钥
    
    Returns:
        List[Dict]: 所有题目的合并列表
    """
    # 检查输出文件是否已存在且完整（在开始处理之前检查）
    output_path = Path(output_file)
    if output_path.exists() and not force:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_questions = json.load(f)
            if existing_questions and len(existing_questions) >= 25:
                # 检查题目编号是否完整（1-25）
                question_numbers = sorted([int(q.get('question','').split('.')[0]) for q in existing_questions 
                                         if q.get('question') and q.get('question','').split('.')[0].isdigit()])
                if question_numbers == list(range(1, 26)):
                    print(f"✓ 发现已存在的完整结果文件: {output_file}")
                    print(f"  包含 {len(existing_questions)} 道题目（编号 1-25）")
                    print(f"  跳过重新解析，直接使用现有结果。")
                    print()
                    return existing_questions
                else:
                    print(f"⚠ 发现结果文件，但题目不完整（编号: {question_numbers}）")
                    print(f"  将重新解析...")
                    print()
        except Exception as e:
            print(f"⚠ 无法读取现有结果文件: {e}")
            print(f"  将重新解析...")
            print()
    elif not output_path.exists():
        print(f"ℹ 输出文件不存在，将开始解析...")
        print()
    
    all_questions = []
    
    # 只处理 Amboss 文件
    amboss_file = Path(directory) / "Amboss Questions.pdf"
    
    if amboss_file.exists():
        pdf_files = [amboss_file]
        print(f"Found Amboss file: {amboss_file.name}")
    else:
        # 如果没有找到精确匹配，尝试模糊匹配
        pdf_files = [f for f in Path(directory).glob("*.pdf") if 'amboss' in f.name.lower()]
        if pdf_files:
            print(f"Found {len(pdf_files)} Amboss file(s)")
        else:
            print(f"Error: No Amboss PDF found in {directory}")
            return []
    
    print()
    
    # 处理 Amboss 文件
    for pdf_file in pdf_files:
        
        # 解析当前文件
        print(f"Processing: {pdf_file.name}")
        questions = parse_pdf_with_openai(str(pdf_file), api_key, model=model, batch_size=batch_size, max_workers=max_workers, overlap_pages=overlap_pages)
        print(f"  Parsed: {len(questions)} questions")
        all_questions.extend(questions)
        print()
    
    # 验证和过滤无效题目
    print("Validating questions...")
    before_validation = len(all_questions)
    all_questions = [q for q in all_questions if validate_question(q)]
    after_validation = len(all_questions)
    invalid_count = before_validation - after_validation
    if invalid_count > 0:
        print(f"  Removed {invalid_count} invalid questions (missing question number or incomplete)")
    print(f"  Valid questions: {after_validation}")
    
    # 去重
    print("Removing duplicates...")
    before_dedup = len(all_questions)
    all_questions = deduplicate_questions(all_questions)
    after_dedup = len(all_questions)
    print(f"  Removed {before_dedup - after_dedup} duplicate questions")
    print(f"  Final count: {after_dedup} unique questions")
    
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
        description='Amboss PDF Parser (OpenAI) - 使用 OpenAI Vision API 解析 Amboss 题库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 解析 Amboss 文件（在目录中自动查找）
  python parse_qbank_openai.py --dir "Qbanks and Practice Exams"
  
  # 解析指定的 Amboss PDF
  python parse_qbank_openai.py "Amboss Questions.pdf"
  
  # 指定输出文件（默认：qbank_amboss_openai.json）
  python parse_qbank_openai.py --dir "Qbanks and Practice Exams" -o custom_output.json
  
  # 指定 API 密钥
  python parse_qbank_openai.py --dir "Qbanks and Practice Exams" --api-key sk-...
        """
    )
    
    # 位置参数：PDF 文件路径（可选）
    parser.add_argument('pdf', nargs='?', help='PDF 文件路径（单个文件）')
    
    # 可选参数：目录
    parser.add_argument('--dir', '-d', help='PDF 文件目录（批量解析）')
    
    # 可选参数：输出文件
    parser.add_argument('--output', '-o', default='qbank_amboss_openai.json', 
                       help='输出 JSON 文件路径（默认：qbank_amboss_openai.json）')
    
    # 可选参数：OpenAI API 密钥
    parser.add_argument('--api-key', '--openai-key',
                       help='OpenAI API 密钥（如果不提供，尝试从环境变量或 api-key.js 读取）')
    
    # 可选参数：OpenAI 模型
    parser.add_argument('--model', '-m', default='gpt-5',
                       help='OpenAI 模型（默认：gpt-5，如果不可用会自动回退到 gpt-4o-2024-11-20。可选：gpt-5, gpt-4o-2024-11-20, gpt-4o, gpt-4o-mini）')
    
    # 可选参数：批量大小（每批处理的页数）
    parser.add_argument('--batch-size', '-b', type=int, default=3,
                       help='每批处理的页数（默认：3=批量处理，设置为 0 为一次性处理整个文件，1 为逐页处理。对于大文档建议使用 3-10）')
    
    # 可选参数：并发线程数
    parser.add_argument('--workers', '-w', type=int, default=3,
                       help='并发处理的线程数（默认：3，设置为 1 禁用并发）')
    
    # 可选参数：重叠页数
    parser.add_argument('--overlap', type=int, default=2,
                       help='批次之间的重叠页数（默认：2，确保跨页题目不丢失）')
    
    # 可选参数：强制重新解析
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制重新解析，即使结果文件已存在且完整')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 显示标题
    print("=" * 60)
    print("Amboss PDF Parser (OpenAI Vision API)")
    print("=" * 60)
    print()
    
    # 根据参数选择处理方式
    if args.dir:
        # 情况 1：指定了目录，批量解析
        parse_directory(args.dir, args.output, args.api_key, args.model, args.batch_size, args.workers, args.overlap, args.force)
    elif args.pdf:
        # 情况 2：指定了单个文件，解析单个文件
        questions = parse_pdf_with_openai(args.pdf, args.api_key, args.model, args.batch_size, args.workers, args.overlap)
        if questions:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"\nSaved {len(questions)} questions to {args.output}")
    else:
        # 情况 3：没有指定参数，尝试默认目录
        qbank_dir = "Qbanks and Practice Exams"
        if os.path.exists(qbank_dir):
            parse_directory(qbank_dir, args.output, args.api_key, args.model, args.batch_size, args.workers, args.overlap, args.force)
        else:
            # 默认目录不存在，显示帮助信息
            parser.print_help()


if __name__ == "__main__":
    main()

