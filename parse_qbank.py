"""
================================================================================
Qbank PDF Parser - 题库 PDF 解析器
================================================================================

功能：
    从 PDF 文件中解析医学考试题目，输出 JSON 格式

支持的 PDF 类型：
    1. Surgery Qbank (NBME 格式) - 文本 PDF
    2. Amboss Qbank - 文本 PDF  
    3. 扫描 PDF - 使用 OCR 解析

输出格式：
    [
        {
            "question": "题目文本...",
            "options": {"A": "选项A", "B": "选项B", ...},
            "correct_answer": "A",
            "explanation": "解释...",
            "source": "来源",
            "source_file": "文件名.pdf",
            "page_number": 1
        },
        ...
    ]

依赖：
    pip install langchain-community pypdf pytesseract pillow pymupdf

OCR 依赖（用于扫描 PDF）：
    - Tesseract OCR: winget install UB-Mannheim.TesseractOCR
    - Poppler: https://github.com/oschwartz10612/poppler-windows/releases

作者：AI for Education
================================================================================
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# =============================================================================
# 依赖检查模块
# =============================================================================
# 
# 功能：检查并导入所有必需的 Python 库
# 
# 为什么需要依赖检查？
#   - 不同功能需要不同的库（文本 PDF vs OCR）
#   - 用户可能没有安装所有依赖
#   - 优雅降级：缺少某些库时仍可使用其他功能
# 
# 依赖库说明：
#   1. PyPDF (langchain-community): 基础 PDF 文本提取
#      - 用于解析文本型 PDF（Surgery, Amboss）
#      - 如果缺失，文本 PDF 解析功能不可用
#   
#   2. PyMuPDF (fitz): 高级 PDF 处理
#      - 用于 OCR 扫描 PDF（将页面渲染为图片）
#      - 比 PyPDF 更快，支持更多格式
#      - 如果缺失，OCR 功能不可用
#   
#   3. Tesseract OCR (pytesseract): 光学字符识别
#      - 用于扫描 PDF 的文字识别
#      - 需要单独安装 Tesseract 可执行文件
#      - 如果缺失，OCR 功能不可用
#   
#   4. PIL/Pillow (Image): 图像处理
#      - 用于 OCR 过程中的图像处理
#      - 如果缺失，OCR 功能不可用
# 
# =============================================================================

def check_dependencies():
    """
    检查并导入所有必需的依赖库。
    
    此函数会尝试导入每个库，如果失败则设置为 None。
    这样其他函数可以检查依赖是否可用，并优雅地处理缺失情况。
    
    Returns:
        dict: 包含所有依赖库的字典
            - deps['pypdf']: PyPDFLoader 类或 None
            - deps['pymupdf']: fitz 模块或 None
            - deps['tesseract']: pytesseract 模块或 None
            - deps['pil']: PIL.Image 类或 None
    
    注意：
        - 如果库未安装，会打印警告信息
        - 不会抛出异常，允许程序继续运行
        - Tesseract 会尝试自动查找可执行文件路径
    """
    deps = {}
    
    # ========================================================================
    # PyPDF (基础 PDF 解析)
    # ========================================================================
    # 用途：从文本型 PDF 中提取文本内容
    # 库：langchain-community 提供的 PyPDFLoader
    # 为什么用 langchain：它封装了 PyPDF2，使用更方便
    try:
        from langchain_community.document_loaders import PyPDFLoader
        deps['pypdf'] = PyPDFLoader
    except ImportError:
        print("Warning: langchain-community not installed. Run: pip install langchain-community pypdf")
        deps['pypdf'] = None
    
    # ========================================================================
    # PyMuPDF (更好的 PDF 解析)
    # ========================================================================
    # 用途：高级 PDF 处理，特别是将页面渲染为图片（用于 OCR）
    # 库：PyMuPDF (导入名为 fitz，这是历史原因)
    # 为什么用 PyMuPDF：比 PyPDF 更快，支持更多格式，可以渲染页面为图片
    try:
        import fitz  # PyMuPDF 的导入名
        deps['pymupdf'] = fitz
    except ImportError:
        print("Warning: PyMuPDF not installed. Run: pip install pymupdf")
        deps['pymupdf'] = None
    
    # ========================================================================
    # Tesseract OCR
    # ========================================================================
    # 用途：光学字符识别，将图片中的文字转换为文本
    # 库：pytesseract (Python 封装) + Tesseract OCR (C++ 可执行文件)
    # 
    # 为什么需要查找可执行文件？
    #   - Tesseract 是独立的 C++ 程序，不是 Python 包
    #   - Windows 上需要指定 tesseract.exe 的完整路径
    #   - Linux/Mac 通常在系统路径中，但也要检查
    try:
        import pytesseract
        from PIL import Image
        
        # 检查 Tesseract 可执行文件
        # Windows 常见安装路径
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",      # 64位系统默认路径
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", # 32位系统路径
            "/usr/bin/tesseract",                                   # Linux 常见路径
            "/usr/local/bin/tesseract",                             # Mac/Linux 本地安装路径
        ]
        
        # 尝试找到 Tesseract 可执行文件
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        
        deps['tesseract'] = pytesseract
        deps['pil'] = Image
    except ImportError:
        print("Warning: pytesseract not installed. Run: pip install pytesseract pillow")
        deps['tesseract'] = None
        deps['pil'] = None
    
    return deps

# 全局依赖字典，在模块加载时初始化
# 其他函数通过检查 DEPS['xxx'] 是否为 None 来判断依赖是否可用
DEPS = check_dependencies()


# =============================================================================
# 工具函数
# =============================================================================
# 
# 这些函数被多个解析器共享，用于文本清理和选项提取
# 
# =============================================================================

def clean_text(text: str) -> str:
    """
    清理从 PDF 提取的文本，移除噪音和格式问题。
    
    PDF 文本提取常见问题：
        1. 多余空白：多个空格、制表符、换行符
        2. PDF 噪音：页眉页脚、导航按钮、链接等
        3. 格式问题：不规范的换行、空格
    
    处理步骤：
        1. 检查空文本
        2. 合并所有空白字符为单个空格
        3. 移除特定 PDF 噪音模式
        4. 去除首尾空白
    
    Args:
        text: 原始文本字符串
    
    Returns:
        str: 清理后的文本
    
    正则表达式说明：
        - r'\s+': 匹配一个或多个空白字符（空格、制表符、换行等）
        - r'https://t\.me/\S+': 匹配 Telegram 链接（PDF 中常见的水印）
        - r'Previous\s+Next.*?Pause': 匹配导航按钮文本（非贪婪匹配）
        - r'Score\s+Report.*?Pause': 匹配分数报告文本
    
    注意：
        此函数会移除所有换行符，适用于解释文本。
        对于包含表格数据的题目文本，请使用 clean_question_text()。
    """
    if not text:
        return ""
    
    # 移除页码标记（用于 Amboss PDF 追踪，但不应出现在最终输出中）
    text = re.sub(r'\[PAGE:\d+\]', '', text)
    
    # 移除多余空白
    # 将所有连续的空白字符（空格、制表符、换行等）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除常见的 PDF 噪音
    # Telegram 链接（常见于某些 PDF 水印）
    text = re.sub(r'https://t\.me/\S+', '', text)
    
    # 导航按钮文本（Previous Next Pause 等）
    # .*? 是非贪婪匹配，匹配到第一个 "Pause" 就停止
    text = re.sub(r'Previous\s+Next.*?Pause', '', text, flags=re.IGNORECASE)
    
    # 分数报告文本
    text = re.sub(r'Score\s+Report.*?Pause', '', text, flags=re.IGNORECASE)
    
    # ========================================================================
    # 移除 PDF 解析错误产生的噪音字符
    # ========================================================================
    # 这些字符通常出现在文本末尾，是 PDF 解析错误导致的
    # 常见模式：
    #   - \" , ~ F' r ,"
    #   - r " , ~ r-- r
    #   - ~ r-- r
    #   - , ~ F' r
    # 
    # 清理策略：
    #   1. 移除末尾的转义引号 + 噪音字符组合
    #   2. 移除末尾的孤立噪音字符（引号、逗号、波浪号、字母 r 等）
    #   3. 移除末尾的噪音模式（~ r-- r 等）
    
    # 移除末尾的转义引号和后续噪音
    text = re.sub(r'\\["\']\s*[,~]\s*[A-Za-z]?["\']?\s*[rR]?\s*[-~]?\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', text)
    
    # 移除末尾的噪音模式：~ r-- r 或 r " , ~ r-- r 等
    text = re.sub(r'\s*[rR]?\s*["\']?\s*[,~]\s*[~-]?\s*[rR]?\s*[-~]+\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', text)
    
    # 移除末尾的孤立噪音字符组合
    # 匹配：逗号、波浪号、引号、字母 r、破折号等的组合
    text = re.sub(r'\s*["\']?\s*[,~]\s*[A-Za-z]?["\']?\s*[rR]\s*[-~]?\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', text)
    
    # 移除末尾的孤立字符（引号、逗号、波浪号、破折号等）
    text = re.sub(r'\s*["\',~-]+\s*$', '', text)
    
    # 移除末尾的单个字母 r（可能是解析错误）
    text = re.sub(r'\s+[rR]\s*$', '', text)
    
    # ========================================================================
    # 移除下划线（PDF 中的填空线）
    # ========================================================================
    # 下划线通常出现在选项末尾，是 PDF 中的填空线或格式标记
    # 例如：E. Venous insufficiency, ____________________________________________________________________________ ___
    # 
    # 清理策略：
    #   1. 移除末尾的连续下划线（3 个或更多）
    #   2. 移除末尾的下划线 + 空格组合
    #   3. 移除文本中间的长下划线（可能是分隔符）
    
    # 移除末尾的连续下划线（3 个或更多）
    text = re.sub(r'[,\s]*_{3,}\s*$', '', text)
    
    # 移除末尾的下划线 + 空格组合
    text = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', text)
    
    # 移除文本末尾的下划线模式（如：, ___ 或 , _ _ _）
    text = re.sub(r'[,\s]+_+\s*$', '', text)
    
    return text.strip()


def clean_question_text(text: str) -> str:
    """
    清理题目文本，保留换行符以维持表格数据格式。
    
    与 clean_text() 的区别：
        - clean_text(): 移除所有换行符，适用于解释文本
        - clean_question_text(): 保留换行符，适用于包含表格数据的题目
    
    处理步骤：
        1. 检查空文本
        2. 清理多个连续空格（但保留换行符）
        3. 移除特定 PDF 噪音模式
        4. 规范化换行符（统一为 \n）
        5. 去除首尾空白
    
    Args:
        text: 原始题目文本字符串
    
    Returns:
        str: 清理后的题目文本（保留换行符）
    
    使用场景：
        - 题目中包含表格数据（如实验室检查结果）
        - 需要保留多行格式的题目
    """
    if not text:
        return ""
    
    # 移除页码标记（用于 Amboss PDF 追踪，但不应出现在最终输出中）
    text = re.sub(r'\[PAGE:\d+\]', '', text)
    
    # 移除常见的 PDF 噪音
    # Telegram 链接（常见于某些 PDF 水印）
    text = re.sub(r'https://t\.me/\S+', '', text)
    
    # 导航按钮文本（Previous Next Pause 等）
    text = re.sub(r'Previous\s+Next.*?Pause', '', text, flags=re.IGNORECASE)
    
    # 分数报告文本
    text = re.sub(r'Score\s+Report.*?Pause', '', text, flags=re.IGNORECASE)
    
    # 规范化换行符：统一为 \n
    # 将 Windows 换行符 (\r\n) 和 Mac 换行符 (\r) 统一为 Unix 换行符 (\n)
    text = re.sub(r'\r\n?', '\n', text)
    
    # 清理多个连续空格（但保留换行符和实验数据的格式）
    # 检测实验数据区域，对实验数据区域进行特殊处理
    lines = text.split('\n')
    cleaned_lines = []
    prev_was_empty = False
    in_lab_section = False  # 是否在实验数据区域
    
    for i, line in enumerate(lines):
        # 检测是否进入实验数据区域
        # 实验数据通常以 "Laboratory studies show:", "Lab studies:", "Laboratory:" 等开始
        lab_markers = [
            r'Laboratory\s+studies\s+show\s*:',
            r'Lab\s+studies\s+show\s*:',
            r'Laboratory\s*:',
            r'Lab\s*:',
            r'Laboratory\s+studies\s*:',
        ]
        
        for marker in lab_markers:
            if re.search(marker, line, re.IGNORECASE):
                in_lab_section = True
                break
        
        # 检测是否离开实验数据区域
        # 如果遇到问题结束标记（如 "Which of the following"），离开实验数据区域
        if in_lab_section:
            question_end_markers = [
                r'Which\s+of\s+the\s+following',
                r'What\s+is',
                r'What\s+are',
                r'What\s+would',
                r'The\s+most\s+likely',
                r'The\s+most\s+appropriate',
                r'Correct\s+Answer\s*:',
            ]
            for marker in question_end_markers:
                if re.search(marker, line, re.IGNORECASE):
                    in_lab_section = False
                    break
        
        # 处理实验数据区域：保留原始格式（不合并空格，保留制表符等）
        if in_lab_section:
            # 在实验数据区域，只清理明显的噪音，保留格式
            # 移除页码标记
            line = re.sub(r'\[PAGE:\d+\]', '', line)
            # 去除行首尾空白（但保留行内的空格和制表符）
            line = line.rstrip()  # 只移除行尾空白，保留行首空白（可能是缩进）
            # 不合并多个空格，保留原始格式
        else:
            # 非实验数据区域，正常清理
            # 将行内的多个空格合并为单个空格
            line = re.sub(r' +', ' ', line)
            # 去除行首尾空白
            line = line.strip()
        
        # 对于空行，保留有意义的空行（用于分隔）
        # 但限制连续空行最多为 1 个
        if not line:
            if not prev_was_empty:
                cleaned_lines.append('')  # 保留单个空行
                prev_was_empty = True
            continue
        
        prev_was_empty = False
        cleaned_lines.append(line)
    
    # 合并行，保留换行符
    result = '\n'.join(cleaned_lines)
    
    # 移除开头和结尾的连续空行，但保留中间的空行
    result = result.strip('\n')
    
    # ========================================================================
    # 移除 PDF 解析错误产生的噪音字符（逐行清理）
    # ========================================================================
    # 这些字符通常出现在文本末尾，是 PDF 解析错误导致的
    # 对每行进行清理，但保留换行符结构
    lines = result.split('\n')
    cleaned_final_lines = []
    for line in lines:
        # 清理每行末尾的噪音字符
        # 移除末尾的转义引号和后续噪音
        line = re.sub(r'\\["\']\s*[,~]\s*[A-Za-z]?["\']?\s*[rR]?\s*[-~]?\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', line)
        # 移除末尾的噪音模式
        line = re.sub(r'\s*[rR]?\s*["\']?\s*[,~]\s*[~-]?\s*[rR]?\s*[-~]+\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', line)
        # 移除末尾的孤立噪音字符组合
        line = re.sub(r'\s*["\']?\s*[,~]\s*[A-Za-z]?["\']?\s*[rR]\s*[-~]?\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', line)
        # 移除末尾的孤立字符
        line = re.sub(r'\s*["\',~-]+\s*$', '', line)
        # 移除末尾的单个字母 r
        line = re.sub(r'\s+[rR]\s*$', '', line)
        # 移除末尾的连续下划线（3 个或更多）
        line = re.sub(r'[,\s]*_{3,}\s*$', '', line)
        # 移除末尾的下划线 + 空格组合
        line = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', line)
        # 移除文本末尾的下划线模式
        line = re.sub(r'[,\s]+_+\s*$', '', line)
        cleaned_final_lines.append(line)
    
    result = '\n'.join(cleaned_final_lines)
    
    return result


def clean_explanation_text(text: str) -> str:
    """
    清理解释文本，保留段落结构以便阅读。
    
    与 clean_text() 的区别：
        - clean_text(): 移除所有换行符，合并为单行
        - clean_explanation_text(): 保留段落结构（双换行符分隔段落）
    
    处理步骤：
        1. 检查空文本
        2. 移除页码标记和噪音
        3. 规范化换行符
        4. 识别段落边界（双换行符或句子结尾）
        5. 保留段落结构
        6. 清理每段内的多余空格
    
    Args:
        text: 原始解释文本字符串
    
    Returns:
        str: 清理后的解释文本（保留段落结构，使用 \n\n 分隔段落）
    
    段落识别规则：
        - 双换行符（\n\n）分隔段落
        - 句子结尾（. ! ?）后跟大写字母可能表示新段落
        - "Incorrect Answers:" 等标记通常表示新段落
    """
    if not text:
        return ""
    
    # 移除页码标记
    text = re.sub(r'\[PAGE:\d+\]', '', text)
    
    # 移除常见的 PDF 噪音
    text = re.sub(r'https://t\.me/\S+', '', text)
    text = re.sub(r'Previous\s+Next.*?Pause', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Score\s+Report.*?Pause', '', text, flags=re.IGNORECASE)
    
    # 规范化换行符
    text = re.sub(r'\r\n?', '\n', text)
    
    # 识别段落标记（这些标记后应该开始新段落）
    paragraph_markers = [
        r'Incorrect\s+Answers?:',
        r'Correct\s+Answer:',
        r'Explanation:',
        r'Rationale:',
        r'Note:',
        r'Key\s+Point:',
        r'Important:'
    ]
    
    # 在段落标记前添加双换行符（如果还没有）
    for marker in paragraph_markers:
        text = re.sub(r'(\S)\s*(' + marker + r')', r'\1\n\n\2', text, flags=re.IGNORECASE)
    
    # 识别句子结尾后跟大写字母的情况（可能是新段落）
    # 但不要破坏缩写（如 "Dr.", "Mr." 等）
    text = re.sub(r'([.!?])\s+([A-Z][a-z])', r'\1\n\n\2', text)
    
    # 清理：将多个连续换行符统一为双换行符（段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 按段落分割
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        # 清理每段内的多余空格（但保留换行符用于列表等）
        # 将多个连续空格合并为单个空格
        para = re.sub(r' +', ' ', para)
        # 移除行首尾空白
        para = para.strip()
        
        # 清理噪音字符
        # 移除末尾的转义引号和后续噪音
        para = re.sub(r'\\["\']\s*[,~]\s*[A-Za-z]?["\']?\s*[rR]?\s*[-~]?\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', para)
        # 移除末尾的噪音模式
        para = re.sub(r'\s*[rR]?\s*["\']?\s*[,~]\s*[~-]?\s*[rR]?\s*[-~]+\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', para)
        # 移除末尾的孤立噪音字符组合
        para = re.sub(r'\s*["\']?\s*[,~]\s*[A-Za-z]?["\']?\s*[rR]\s*[-~]?\s*[rR]?\s*["\']?\s*[,~]?\s*["\']?\s*$', '', para)
        # 移除末尾的孤立字符
        para = re.sub(r'\s*["\',~-]+\s*$', '', para)
        # 移除末尾的单个字母 r
        para = re.sub(r'\s+[rR]\s*$', '', para)
        # 移除末尾的下划线
        para = re.sub(r'[,\s]*_{3,}\s*$', '', para)
        para = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', para)
        para = re.sub(r'[,\s]+_+\s*$', '', para)
        
        if para:  # 只保留非空段落
            cleaned_paragraphs.append(para)
    
    # 合并段落，使用双换行符分隔
    result = '\n\n'.join(cleaned_paragraphs)
    
    return result.strip()


def format_lab_data(text: str) -> str:
    """
    格式化实验数据，使其更清晰易读。
    
    功能：
        1. 检测实验数据区域
        2. 规范化实验数据格式（对齐、间距）
        3. 确保实验数据项目之间有适当的换行
        4. 统一实验数据的显示格式
    
    Args:
        text: 包含实验数据的文本
    
    Returns:
        str: 格式化后的文本
    
    格式化规则：
        1. 实验数据项目格式：项目名: 数值 单位
        2. 如果项目名和数值在同一行，确保有适当的间距
        3. 如果实验数据被合并成一行，尝试重新分割
        4. 统一冒号后的空格（": "）
        5. 确保数值和单位之间有空格
    """
    if not text:
        return text
    
    # 检测实验数据的关键词
    lab_keywords = [
        r'Laboratory\s+studies\s+show\s*:',
        r'Lab\s+studies\s+show\s*:',
        r'Laboratory\s*:',
        r'Lab\s*:',
        r'Laboratory\s+studies\s*:',
        r'Laboratory\s+values\s*:',
        r'Laboratory\s+results\s*:',
        r'Lab\s+results\s*:',
        r'Laboratory\s+data\s*:',
        r'Lab\s+values\s*:',
        r'Lab\s+data\s*:',
    ]
    
    # 实验数据单位模式
    lab_units = r'(g/dL|mg/dL|mEq/L|U/L|mm3|/mm3|°C|°F|mmHg|/hpf|ng/mL|pg/cell|µm|kg/m2|/L|/dL|/mL|IU/L|mEq|mmol/L|mg/L|µg/dL|pg/mL|IU/mL|mIU/mL|%|cells/µL|cells/mm3)'
    
    # 实验数据项目名（常见）
    lab_items = [
        'Hemoglobin', 'Hematocrit', 'Leukocyte', 'Platelet', 'WBC', 'RBC',
        'Na+', 'K+', 'Cl-', 'HCO3-', 'Creatinine', 'Glucose', 'Urea',
        'Alkaline phosphatase', 'AST', 'ALT', 'Amylase', 'Lipase',
        'Troponin', 'LDH', 'MCV', 'MCH', 'BUN', 'Bilirubin',
        'Albumin', 'Total protein', 'Calcium', 'Phosphorus', 'Magnesium',
        'PT', 'PTT', 'INR', 'ESR', 'CRP', 'TSH', 'T4', 'T3'
    ]
    
    lines = text.split('\n')
    formatted_lines = []
    in_lab_section = False
    lab_section_start = -1
    
    for i, line in enumerate(lines):
        # 检测是否进入实验数据区域
        for keyword in lab_keywords:
            if re.search(keyword, line, re.IGNORECASE):
                in_lab_section = True
                lab_section_start = i
                break
        
        # 如果在实验数据区域，进行格式化
        if in_lab_section:
            # 检测是否离开实验数据区域
            question_end_markers = [
                r'Which\s+of\s+the\s+following',
                r'What\s+is',
                r'What\s+are',
                r'What\s+would',
                r'The\s+most\s+likely',
                r'The\s+most\s+appropriate',
                r'Correct\s+Answer\s*:',
                r'^[A-J]\)',  # 选项开始
            ]
            
            should_exit = False
            for marker in question_end_markers:
                if re.search(marker, line, re.IGNORECASE):
                    in_lab_section = False
                    should_exit = True
                    break
            
            if should_exit:
                formatted_lines.append(line)
                continue
            
            # 格式化实验数据行
            formatted_line = format_lab_data_line(line, lab_units, lab_items)
            formatted_lines.append(formatted_line)
        else:
            formatted_lines.append(line)
    
    # 如果整个文本都在实验数据区域，尝试整体格式化
    if in_lab_section and lab_section_start >= 0:
        # 合并实验数据部分，尝试重新格式化
        lab_text = '\n'.join(formatted_lines[lab_section_start:])
        formatted_lab = format_lab_data_block(lab_text, lab_units, lab_items)
        formatted_lines = formatted_lines[:lab_section_start] + [formatted_lab]
    
    return '\n'.join(formatted_lines)


def format_lab_data_line(line: str, lab_units: str, lab_items: List[str]) -> str:
    """
    格式化单行实验数据。
    
    Args:
        line: 实验数据行
        lab_units: 实验单位正则模式
        lab_items: 实验项目名列表
    
    Returns:
        str: 格式化后的行
    """
    if not line.strip():
        return line
    
    # 清理行首尾空白
    line = line.strip()
    
    # 规范化冒号后的空格（": "）
    line = re.sub(r':\s*', ': ', line)
    
    # 确保数值和单位之间有空格
    # 例如："14.0g/dL" -> "14.0 g/dL"
    line = re.sub(r'(\d+\.?\d*)\s*(' + lab_units + r')', r'\1 \2', line, flags=re.IGNORECASE)
    
    # 如果行中包含多个实验数据项目（用逗号或分号分隔），尝试分割
    if ',' in line or ';' in line:
        # 检测是否包含多个实验数据项目
        # 例如："Hemoglobin: 14.0 g/dL, Hematocrit: 42%"
        parts = re.split(r'[,;]', line)
        if len(parts) > 1:
            # 检查每个部分是否包含实验数据
            formatted_parts = []
            for part in parts:
                part = part.strip()
                if part:
                    # 格式化每个部分
                    part = re.sub(r':\s*', ': ', part)
                    part = re.sub(r'(\d+\.?\d*)\s*(' + lab_units + r')', r'\1 \2', part, flags=re.IGNORECASE)
                    formatted_parts.append(part)
            
            # 如果格式化成功，用换行符连接（更清晰）
            if len(formatted_parts) > 1:
                return '\n'.join(formatted_parts)
    
    return line


def format_lab_data_block(block: str, lab_units: str, lab_items: List[str]) -> str:
    """
    格式化整个实验数据块（可能被合并成一行）。
    
    Args:
        block: 实验数据块文本
        lab_units: 实验单位正则模式
        lab_items: 实验项目名列表
    
    Returns:
        str: 格式化后的文本块
    """
    if not block.strip():
        return block
    
    # 如果已经包含换行符，按行处理
    if '\n' in block:
        lines = block.split('\n')
        formatted_lines = [format_lab_data_line(line, lab_units, lab_items) for line in lines]
        return '\n'.join(formatted_lines)
    
    # 如果没有换行符，尝试识别实验数据项目并分割
    # 模式1：项目名: 数值 单位
    # 模式2：项目名 数值 单位（无冒号）
    
    # 尝试按常见模式分割
    # 查找所有可能的实验数据项目
    items = []
    
    # 方法1：查找 "项目名: 数值 单位" 模式
    pattern1 = r'([A-Z][a-zA-Z\s\+\-]+?)\s*:\s*(\d+\.?\d*)\s*(' + lab_units + r')'
    matches1 = re.finditer(pattern1, block, re.IGNORECASE)
    for match in matches1:
        item_name = match.group(1).strip()
        value = match.group(2)
        unit = match.group(3)
        items.append(f"{item_name}: {value} {unit}")
    
    # 方法2：查找 "项目名 数值 单位" 模式（无冒号）
    if not items:
        pattern2 = r'([A-Z][a-zA-Z\s\+\-]+?)\s+(\d+\.?\d*)\s*(' + lab_units + r')'
        matches2 = re.finditer(pattern2, block, re.IGNORECASE)
        for match in matches2:
            item_name = match.group(1).strip()
            value = match.group(2)
            unit = match.group(3)
            items.append(f"{item_name}: {value} {unit}")
    
    # 如果找到了实验数据项目，用换行符连接
    if items:
        return '\n'.join(items)
    
    # 如果无法识别，返回原文本
    return block


def parse_option_explanations(explanation_text: str, options: Dict[str, str], correct_answer: str) -> Dict[str, str]:
    """
    从解释文本中解析每个选项对应的解释。
    
    AMBOSS PDF 的解释格式通常为：
    - 整体解释（正确答案的详细说明）
    - "Incorrect Answers: A, B, C, D" 标记
    - 每个错误选项的简短说明
    
    此函数尝试：
    1. 识别正确答案的解释（通常是主要解释）
    2. 识别错误选项的解释（通常在 "Incorrect Answers" 部分）
    3. 为每个选项提取对应的解释文本
    
    Args:
        explanation_text: 完整的解释文本
        options: 选项字典 {"A": "选项文本", ...}
        correct_answer: 正确答案字母
    
    Returns:
        Dict[str, str]: 每个选项对应的解释，格式为 {"A": "解释A", "B": "解释B", ...}
    """
    option_explanations = {}
    
    if not explanation_text or not options:
        return option_explanations
    
    # 规范化换行符
    explanation_text = re.sub(r'\r\n?', '\n', explanation_text)
    
    # 步骤1：提取正确答案的主要解释（解释的开头部分，通常在 "Incorrect Answers" 之前）
    incorrect_markers = [
        r'Incorrect\s+Answers?\s*:',
        r'Incorrect\s+Choices?\s*:',
        r'Other\s+Answers?\s*:',
    ]
    
    incorrect_match = None
    incorrect_pos = len(explanation_text)
    for marker in incorrect_markers:
        match = re.search(marker, explanation_text, re.IGNORECASE)
        if match and match.start() < incorrect_pos:
            incorrect_match = match
            incorrect_pos = match.start()
    
    # 正确答案的解释通常是 "Incorrect Answers" 之前的部分
    if incorrect_match:
        correct_explanation = explanation_text[:incorrect_match.start()].strip()
        incorrect_section = explanation_text[incorrect_match.end():].strip()
        
        # 为主要解释清理文本
        correct_explanation = clean_explanation_text(correct_explanation)
        if correct_explanation and correct_answer:
            option_explanations[correct_answer] = correct_explanation
    else:
        # 如果没有 "Incorrect Answers" 标记，整个解释都是正确答案的解释
        correct_explanation = clean_explanation_text(explanation_text)
        if correct_explanation and correct_answer:
            option_explanations[correct_answer] = correct_explanation
        incorrect_section = ""
    
    # 步骤2：从 "Incorrect Answers" 部分提取每个错误选项的解释
    if incorrect_section:
        # 查找所有选项字母的模式（如 "A, B, C" 或 "Choice A"）
        # 提取错误选项列表
        incorrect_options_text = ""
        incorrect_options_match = re.match(r'([A-J](?:\s*,\s*[A-J])*)', incorrect_section)
        if incorrect_options_match:
            incorrect_options_text = incorrect_options_match.group(1)
            remaining_text = incorrect_section[len(incorrect_options_match.group(0)):].strip()
            
            # 移除开头的句号或冒号
            remaining_text = re.sub(r'^[\.:]\s*', '', remaining_text)
        else:
            remaining_text = incorrect_section
        
        # 尝试按选项字母分割解释文本
        # 查找 "Choice A", "Option A", 或者选项字母开头的段落
        option_letter_pattern = r'(?:Choice|Option)\s+([A-J])(?:\s*[\)\.:])?'
        
        # 按段落分割（双换行符）
        paragraphs = re.split(r'\n\n+', remaining_text)
        
        current_option = None
        current_explanation = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 检查是否开始新的选项解释
            letter_match = re.match(option_letter_pattern, para, re.IGNORECASE)
            if letter_match:
                # 保存前一个选项的解释
                if current_option and current_explanation:
                    opt_expl = clean_text(' '.join(current_explanation))
                    if opt_expl:
                        option_explanations[current_option] = opt_expl
                
                # 开始新选项
                current_option = letter_match.group(1).upper()
                # 移除选项标记，保留解释文本
                remaining = para[len(letter_match.group(0)):].strip()
                remaining = re.sub(r'^[\.:\)]\s*', '', remaining)
                current_explanation = [remaining] if remaining else []
            else:
                # 检查是否以选项字母开头（可能是简化格式）
                letter_match = re.match(r'^([A-J])[\)\.:]\s+', para)
                if letter_match:
                    if current_option and current_explanation:
                        opt_expl = clean_text(' '.join(current_explanation))
                        if opt_expl:
                            option_explanations[current_option] = opt_expl
                    
                    current_option = letter_match.group(1).upper()
                    remaining = para[len(letter_match.group(0)):].strip()
                    current_explanation = [remaining] if remaining else []
                else:
                    # 继续当前选项的解释
                    if current_option:
                        current_explanation.append(para)
                    else:
                        # 如果没有明确的选项标记，可能是整体解释的一部分
                        # 添加到所有未分配的选项
                        pass
        
        # 保存最后一个选项的解释
        if current_option and current_explanation:
            opt_expl = clean_text(' '.join(current_explanation))
            if opt_expl:
                option_explanations[current_option] = opt_expl
        
        # 如果没有找到明确的选项分割，尝试智能分割
        if not option_explanations and remaining_text:
            # 尝试按句子分割，寻找提及选项的部分
            sentences = re.split(r'[.!?]\s+', remaining_text)
            for sentence in sentences:
                # 查找 "Choice A", "Option A" 等
                choice_match = re.search(r'(?:Choice|Option)\s+([A-J])[\)\.:]?\s+(.+)', sentence, re.IGNORECASE)
                if choice_match:
                    opt_letter = choice_match.group(1).upper()
                    opt_expl = choice_match.group(2).strip()
                    if opt_expl:
                        option_explanations[opt_letter] = clean_text(opt_expl)
    
    # 步骤3：为没有解释的选项添加默认说明
    for letter in options.keys():
        if letter not in option_explanations:
            if letter == correct_answer:
                # 正确答案但没有解释，使用整体解释
                if explanation_text:
                    option_explanations[letter] = clean_explanation_text(explanation_text)
            else:
                # 错误选项但没有解释，添加简短说明
                option_explanations[letter] = "This option is incorrect."
    
    return option_explanations


def extract_options(text: str) -> Dict[str, str]:
    """
    从文本中提取选择题选项（A-J，最多10个选项）。
    
    支持的选项格式：
        - "A) 选项文本"
        - "A. 选项文本"
        - "A 选项文本"（较少见）
    
    处理逻辑：
        1. 按行分割文本
        2. 每行检查是否匹配选项格式
        3. 提取选项字母和文本
        4. 过滤太短的选项（可能是误匹配）
    
    Args:
        text: 包含选项的文本
    
    Returns:
        Dict[str, str]: 选项字典，格式为 {"A": "选项文本", "B": "选项文本", ..., "J": "选项文本"}
    
    正则表达式说明：
        - r'^([A-J])[)\.]?\s*(.+)$'
          - ^: 行首
          - ([A-J]): 捕获组1，匹配字母 A-J（支持最多10个选项）
          - [)\.]?: 可选的分隔符（右括号或点号）
          - \s*: 零个或多个空白字符
          - (.+): 捕获组2，匹配选项文本（至少一个字符）
          - $: 行尾
    
    注意：
        - 支持 A-J 的选项（最多10个选项）
        - 选项文本长度必须 > 3 字符（过滤误匹配）
        - 如果同一字母出现多次，后面的会覆盖前面的
    """
    options = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 匹配选项格式：A) text 或 A. text 或 A text
        # 正则说明：
        #   ^([A-J])      - 行首的字母 A-J（捕获组1，支持最多10个选项）
        #   [)\.]?        - 可选的分隔符（右括号或点号）
        #   \s*           - 零个或多个空白字符
        #   (.+)$         - 选项文本直到行尾（捕获组2）
        match = re.match(r'^([A-J])[)\.]?\s*(.+)$', line)
        
        if match:
            letter = match.group(1).upper()  # 选项字母（A-J）
            opt_text = match.group(2).strip()  # 选项文本
            
            # 清理选项文本中的下划线（PDF 中的填空线）
            # 移除末尾的连续下划线（3 个或更多）
            opt_text = re.sub(r'[,\s]*_{3,}\s*$', '', opt_text)
            # 移除末尾的下划线 + 空格组合
            opt_text = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', opt_text)
            # 移除文本末尾的下划线模式
            opt_text = re.sub(r'[,\s]+_+\s*$', '', opt_text)
            # 移除文本中间的长下划线（可能是分隔符，但保留单个下划线用于正常文本）
            opt_text = re.sub(r'_{4,}', '', opt_text)  # 移除 4 个或更多连续下划线
            
            opt_text = opt_text.strip()
            
            # 过滤太短的选项（可能是误匹配）
            if len(opt_text) > 3:
                options[letter] = opt_text
    
    return options


def extract_images_from_page(pdf_path: str, page_num: int, output_dir: str = "images", 
                            filter_by_position: bool = True) -> List[str]:
    """
    从 PDF 的指定页面提取图片，使用智能方法识别解题相关的图片。
    
    功能：
        1. 使用 PyMuPDF 打开 PDF
        2. 分析页面文本布局，识别题目区域
        3. 获取指定页面的所有图片及其位置
        4. 使用多种方法过滤掉页眉、页脚、水印等无关图片：
           - 基于文本上下文分析（图片周围的文本内容）
           - 基于实际文本分布（题目区域的实际边界）
           - 基于图片特征（大小、宽高比、位置）
        5. 只保留与题目相关的图片
        6. 将图片保存到本地目录
        7. 返回图片路径列表
    
    Args:
        pdf_path: PDF 文件路径
        page_num: 页码（从 1 开始）
        output_dir: 图片保存目录（默认：'images'）
        filter_by_position: 是否根据位置和上下文过滤图片（默认：True）
                           - True: 使用智能过滤（推荐）
                           - False: 提取所有图片（不推荐）
    
    Returns:
        List[str]: 图片路径列表（相对于工作目录的路径）
                   如果提取失败或没有图片，返回空列表
    
    图片命名格式：
        {pdf_filename}_{page_num}_{image_index}.png
        例如：Surgery_3_Answers_1_0.png
    
    智能过滤策略：
        1. 尺寸过滤：< 100x100 像素的图片（可能是图标或水印）
        2. 文本上下文分析：
           - 分析图片周围的文本内容
           - 如果图片附近有题目关键词（如选项 A)、B)、问题文本等），保留
           - 如果图片附近只有页眉页脚文本，过滤
        3. 文本分布分析：
           - 识别页面中实际文本的分布区域
           - 只保留文本区域内的图片（不依赖固定边距）
        4. 图片特征过滤：
           - 宽高比异常（< 0.2 或 > 5.0）的图片
           - 过大的图片（> 80% 页面面积）
    
    注意事项：
        - 需要 PyMuPDF (fitz) 库
        - 如果图片目录不存在，会自动创建
        - 智能过滤适用于各种布局的 PDF（图片在中间、右侧等）
    """
    if not DEPS['pymupdf']:
        return []  # PyMuPDF 不可用，返回空列表
    
    fitz = DEPS['pymupdf']
    
    try:
        # 打开 PDF 文件
        doc = fitz.open(pdf_path)
        
        # 检查页码有效性（page_num 从 1 开始，转换为 0 开始的索引）
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return []
        
        # 获取页面对象（转换为 0 开始的索引）
        page = doc[page_num - 1]
        
        # 获取页面尺寸（用于位置过滤）
        page_rect = page.rect  # 页面矩形 (x0, y0, x1, y1)
        page_width = page_rect.width
        page_height = page_rect.height
        
        # ========================================================================
        # 步骤 1：分析页面文本布局，识别题目区域的实际边界
        # ========================================================================
        # 获取页面中的所有文本块及其位置
        text_blocks = page.get_text("dict")["blocks"]
        
        # 识别题目相关的文本区域
        # 题目文本的特征：
        #   - 包含选项标记（A)、B)、C) 等）
        #   - 包含问题关键词（"year-old", "patient", "diagnosis" 等）
        #   - 包含 "Correct Answer:" 标记
        question_keywords = [
            r'^[A-J]\)',  # 选项标记
            r'\d+\s*-?\s*year\s*-?\s*old',  # 年龄描述
            r'patient', r'diagnosis', r'treatment', r'symptom',
            r'Correct\s+Answer', r'Laboratory\s+studies'
        ]
        
        # 计算文本区域的实际边界（不依赖固定边距）
        text_x0 = page_width  # 最小 x 坐标
        text_y0 = page_height  # 最小 y 坐标
        text_x1 = 0  # 最大 x 坐标
        text_y1 = 0  # 最大 y 坐标
        
        has_question_content = False
        
        for block in text_blocks:
            if "lines" not in block:
                continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    # 检查是否包含题目相关内容
                    is_question_text = False
                    for keyword in question_keywords:
                        if re.search(keyword, text, re.IGNORECASE):
                            is_question_text = True
                            has_question_content = True
                            break
                    
                    # 如果包含题目内容，更新文本区域边界
                    if is_question_text or len(text) > 20:  # 长文本也可能是题目内容
                        bbox = span.get("bbox", [0, 0, 0, 0])  # [x0, y0, x1, y1]
                        if len(bbox) == 4:
                            text_x0 = min(text_x0, bbox[0])
                            text_y0 = min(text_y0, bbox[1])
                            text_x1 = max(text_x1, bbox[2])
                            text_y1 = max(text_y1, bbox[3])
        
        # 如果找到了题目内容，使用实际文本区域；否则使用保守的默认区域
        if has_question_content and text_x1 > text_x0 and text_y1 > text_y0:
            # 扩展文本区域边界（给图片一些容差）
            margin_x = (text_x1 - text_x0) * 0.1  # 10% 的水平容差
            margin_y = (text_y1 - text_y0) * 0.1  # 10% 的垂直容差
            
            content_area = {
                'x0': max(0, text_x0 - margin_x),
                'y0': max(0, text_y0 - margin_y),
                'x1': min(page_width, text_x1 + margin_x),
                'y1': min(page_height, text_y1 + margin_y)
            }
        else:
            # 如果没有找到题目内容，使用保守的默认区域（排除明显的页眉页脚）
            top_margin = page_height * 0.10  # 减少到 10%
            bottom_margin = page_height * 0.10
            side_margin = page_width * 0.05  # 减少到 5%（允许右侧图片）
            
            content_area = {
                'x0': side_margin,
                'y0': top_margin,
                'x1': page_width - side_margin,
                'y1': page_height - bottom_margin
            }
        
        # 获取页面中的所有图片及其位置
        image_list = page.get_images()
        
        if not image_list:
            doc.close()
            return []  # 没有图片
        
        # 创建图片保存目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成基础文件名（不含扩展名）
        pdf_name = Path(pdf_path).stem  # 获取文件名（不含扩展名）
        # 清理文件名中的特殊字符，避免文件系统问题
        pdf_name = re.sub(r'[<>:"/\\|?*]', '_', pdf_name)
        
        # 先收集所有通过传统过滤的图片信息
        candidate_images = []  # 存储候选图片信息
        image_paths = []
        
        # 提取每张图片
        for img_index, img in enumerate(image_list):
            try:
                # 获取图片的 xref（交叉引用）
                xref = img[0]
                
                # 提取图片数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # 图片扩展名（如 'png', 'jpeg'）
                
                # 获取图片尺寸
                # 如果图片太小（可能是图标或水印），跳过
                if "width" in base_image and "height" in base_image:
                    width = base_image["width"]
                    height = base_image["height"]
                    # 过滤太小的图片（< 100x100 像素）
                    if width < 100 or height < 100:
                        continue
                
                # ================================================================
                # 智能过滤：基于位置和文本上下文分析
                # ================================================================
                if filter_by_position:
                    # 获取图片在页面上的位置
                    image_rects = page.get_image_rects(xref)
                    
                    if not image_rects:
                        # 如果无法获取位置，跳过（可能是背景图片或水印）
                        continue
                    
                    # 获取图片的主要位置（使用第一个矩形）
                    img_rect = image_rects[0]
                    img_center_x = (img_rect.x0 + img_rect.x1) / 2
                    img_center_y = (img_rect.y0 + img_rect.y1) / 2
                    img_area = (img_rect.x1 - img_rect.x0) * (img_rect.y1 - img_rect.y0)
                    
                    # ============================================================
                    # 方法 1：检查图片是否在文本区域内
                    # ============================================================
                    is_in_content_area = False
                    
                    # 检查图片中心点是否在文本区域内
                    if (content_area['x0'] <= img_center_x <= content_area['x1'] and
                        content_area['y0'] <= img_center_y <= content_area['y1']):
                        is_in_content_area = True
                    
                    # 或者检查图片的主要部分（至少 40%）是否在文本区域内
                    if not is_in_content_area:
                        overlap_x0 = max(img_rect.x0, content_area['x0'])
                        overlap_y0 = max(img_rect.y0, content_area['y0'])
                        overlap_x1 = min(img_rect.x1, content_area['x1'])
                        overlap_y1 = min(img_rect.y1, content_area['y1'])
                        
                        if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
                            overlap_area = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
                            if overlap_area >= img_area * 0.4:  # 至少 40% 重叠
                                is_in_content_area = True
                    
                    # ============================================================
                    # 方法 2：分析图片周围的文本上下文（精准模式，保持完整分析）
                    # ============================================================
                    # 定义图片周围的文本搜索区域（扩展图片边界）
                    search_margin = min(page_width, page_height) * 0.05  # 5% 的搜索范围
                    search_area = {
                        'x0': max(0, img_rect.x0 - search_margin),
                        'y0': max(0, img_rect.y0 - search_margin),
                        'x1': min(page_width, img_rect.x1 + search_margin),
                        'y1': min(page_height, img_rect.y1 + search_margin)
                    }
                    
                    # 在图片周围搜索题目相关的文本
                    nearby_question_text = False
                    nearby_header_footer = False
                    
                    for block in text_blocks:
                        if "lines" not in block:
                            continue
                        
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                if len(bbox) != 4:
                                    continue
                                
                                # 检查文本是否在图片附近
                                span_center_x = (bbox[0] + bbox[2]) / 2
                                span_center_y = (bbox[1] + bbox[3]) / 2
                                
                                if (search_area['x0'] <= span_center_x <= search_area['x1'] and
                                    search_area['y0'] <= span_center_y <= search_area['y1']):
                                    text = span.get("text", "").strip()
                                    
                                    # 检查是否是题目相关文本
                                    for keyword in question_keywords:
                                        if re.search(keyword, text, re.IGNORECASE):
                                            nearby_question_text = True
                                            break
                                    
                                    # 检查是否是页眉页脚文本
                                    header_footer_keywords = [
                                        r'Page\s+\d+', r'^\d+$',  # 页码
                                        r'Exam\s+Section', r'Score\s+Report',  # 页眉
                                        r'Previous', r'Next', r'Pause'  # 导航按钮
                                    ]
                                    for keyword in header_footer_keywords:
                                        if re.search(keyword, text, re.IGNORECASE):
                                            nearby_header_footer = True
                                            break
                                    
                                    if nearby_question_text:
                                        break
                            
                            if nearby_question_text:
                                break
                        
                        if nearby_question_text:
                            break
                    
                    # ============================================================
                    # 综合判断：是否保留图片（极宽松版本，减少漏图）
                    # ============================================================
                    # 保留策略：默认保留，只过滤明显的页眉页脚
                    # 这样可以最大程度减少漏图，即使可能包含一些噪音图片
                    should_keep = True
                    
                    # 只过滤明显的页眉页脚区域（缩小到顶部5%和底部5%）
                    top_margin = page_height * 0.05  # 顶部 5%（更宽松）
                    bottom_margin = page_height * 0.05  # 底部 5%（更宽松）
                    is_in_header_footer = (img_center_y < top_margin or img_center_y > page_height - bottom_margin)
                    
                    # 如果图片在明显的页眉页脚区域，且附近有页眉页脚文本，才过滤
                    if is_in_header_footer and nearby_header_footer:
                        # 进一步检查：如果附近也有题目文本，保留（可能是题目图片在顶部）
                        if not nearby_question_text:
                            should_keep = False
                    
                    if not should_keep:
                        # 不满足保留条件，跳过
                        continue
                
                # ================================================================
                # 额外过滤：过滤掉宽高比异常的图片（极宽松版本）
                # ================================================================
                # 医学图像可能有各种宽高比，包括非常极端的情况
                # 只过滤明显异常的图片（如单像素线条）
                aspect_ratio = width / height if height > 0 else 0
                # 极宽松的宽高比限制：0.01-100.0（几乎不过滤）
                # 只过滤明显是装饰线条的图片（单像素宽或高）
                if aspect_ratio < 0.01 or aspect_ratio > 100.0:
                    # 宽高比极端异常，可能是单像素线条，跳过
                    continue
                
                # ================================================================
                # 额外过滤：过滤掉过大的图片（可能是背景或整页截图）
                # ================================================================
                # 如果图片占页面面积超过 95%，可能是背景图片（提高阈值）
                if image_rects:
                    for rect in image_rects:
                        img_page_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                        page_area = page_width * page_height
                        if img_page_area > page_area * 0.95:
                            # 图片几乎占满整个页面，可能是背景，跳过
                            continue
                        break
                
                # ================================================================
                # 文字检测：过滤掉包含大量文字的图片（改进版，更宽松，可选）
                # ================================================================
                # 解题图片应该是纯医学图像（X光片、CT扫描等）
                # 医学图像可能包含少量标注文字（如箭头、标签），这是正常的
                # 只有包含大量文字（如整页截图）的图片才应该被过滤
                # 
                # 注意：如果图片提取失败或过滤太严格，可以暂时禁用文字检测
                # 通过设置 skip_text_detection = True 来跳过文字检测
                skip_text_detection = True  # 暂时设置为 True，跳过文字检测以确保图片被提取
                
                if not skip_text_detection:
                    # 方法 1：快速启发式检测（基于图片特征）
                    # ============================================================
                    # 如果图片宽高比接近页面宽高比，可能是整页截图
                    page_aspect_ratio = page_width / page_height if page_height > 0 else 0
                    img_aspect_ratio = width / height if height > 0 else 0
                    
                    is_likely_text_image = False
                    
                    # 检查 1：如果图片宽高比与页面宽高比非常接近（误差 < 2%），且面积很大，可能是整页截图
                    if page_aspect_ratio > 0 and abs(img_aspect_ratio - page_aspect_ratio) / page_aspect_ratio < 0.02:
                        # 进一步检查：如果图片大小接近页面大小，很可能是整页截图
                        if image_rects:
                            for rect in image_rects:
                                img_page_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                                page_area = page_width * page_height
                                # 如果图片占页面面积 > 80%（进一步提高阈值），且宽高比接近，可能是整页截图
                                if img_page_area > page_area * 0.80:
                                    is_likely_text_image = True
                                    break
                    
                    # 方法 2：OCR 文字检测（仅在启发式检测不确定时使用，且更宽松）
                    # ============================================================
                    # 对于 Surgery PDF，医学图像可能包含少量标注，不应该被过滤
                    # 只有包含大量文字（如整页文本）的图片才应该被过滤
                    # 注意：OCR 检测较慢，可以暂时禁用以提高速度
                    use_ocr_detection = False  # 设置为 False 可以跳过 OCR 检测
                    
                    if not is_likely_text_image and use_ocr_detection and DEPS['pil'] and DEPS['tesseract']:
                        try:
                            from io import BytesIO
                            Image = DEPS['pil']
                            pytesseract = DEPS['tesseract']
                            
                            # 将图片字节转换为 PIL Image 对象
                            img = Image.open(BytesIO(image_bytes))
                            
                            # 使用 OCR 检测图片中的文字
                            # 使用快速模式：只检测文字密度，不提取完整文本
                            try:
                                # 使用 OCR 的快速模式，只检测是否有大量文字
                                # 限制字符集，提高速度
                                ocr_text = pytesseract.image_to_string(
                                    img, 
                                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}'
                                )
                                
                                # 计算文字密度（文字字符数 / 图片像素数）
                                text_length = len(ocr_text.strip())
                                pixel_count = width * height
                                
                                # 如果文字密度超过阈值，可能是文字图片
                                if pixel_count > 0:
                                    text_density = text_length / pixel_count
                                    
                                    # 非常宽松的阈值判断：
                                    #   1. 文字密度 > 0.003（每 333 像素至少 1 个字符，进一步提高阈值）
                                    #   2. 文字长度 > 200 字符（进一步提高阈值，排除少量标注）
                                    if text_density > 0.003 and text_length > 200:
                                        # 进一步检查：如果包含大量常见单词，更可能是文字图片
                                        common_words = [
                                            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                                            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 
                                            'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 
                                            'way', 'who', 'with', 'this', 'that', 'from', 'they', 'have', 
                                            'been', 'what', 'when', 'where', 'which', 'would', 'could', 
                                            'should', 'about', 'after', 'before', 'during', 'patient', 
                                            'diagnosis', 'treatment', 'symptom', 'disease', 'medical',
                                            'examination', 'history', 'physical', 'laboratory', 'studies'
                                        ]
                                        text_lower = ocr_text.lower()
                                        word_count = sum(1 for word in common_words if word in text_lower)
                                        
                                        # 如果包含 8 个或更多常见单词（进一步提高阈值），很可能是文字图片
                                        if word_count >= 8:
                                            is_likely_text_image = True
                                        
                                        # 如果文字长度 > 500 字符（进一步提高阈值），也很可能是文字图片
                                        elif text_length > 500:
                                            is_likely_text_image = True
                            except Exception as ocr_error:
                                # OCR 失败，可能是图片质量问题，继续处理
                                # 不跳过，因为可能是有效的医学图像
                                pass
                        except Exception as img_error:
                            # 图片处理失败，继续处理下一张
                            pass
                    
                    # 如果检测到是文字图片，跳过
                    if is_likely_text_image:
                        continue
                
                # 收集候选图片信息
                candidate_images.append({
                    'index': len(candidate_images),  # 在候选列表中的索引
                    'img_index': img_index,  # 原始图片索引
                    'image_bytes': image_bytes,
                    'image_ext': image_ext,
                    'rect': img_rect if image_rects else None
                })
                
            except Exception as e:
                # 如果某张图片提取失败，继续处理下一张
                print(f"    Warning: Failed to extract image {img_index} from page {page_num}: {e}")
                continue
        
        # ========================================================================
        # 保存最终选定的图片
        # ========================================================================
        for candidate in candidate_images:
            try:
                # 生成图片文件名
                image_filename = f"{pdf_name}_{page_num}_{candidate['img_index']}.{candidate['image_ext']}"
                image_path = output_path / image_filename
                
                # 保存图片
                with open(image_path, "wb") as img_file:
                    img_file.write(candidate['image_bytes'])
                
                # 保存相对路径（相对于工作目录）
                relative_path = str(image_path)
                image_paths.append(relative_path)
            except Exception as e:
                print(f"    Warning: Failed to save image {candidate['img_index']}: {e}")
                continue
        
        doc.close()
        return image_paths
        
    except Exception as e:
        print(f"  Error extracting images from page {page_num}: {e}")
        return []


# =============================================================================
# Surgery PDF 解析器 (文本 PDF)
# =============================================================================
# 
# 功能：解析 Surgery Qbank PDF 文件（NBME 格式）
# 
# PDF 格式特征：
#   - 每页通常包含一个完整的题目
#   - 题目以特殊标记开始（■ 或 'I）
#   - 选项格式：A) 选项文本
#   - 正确答案标记：Correct Answer: X
#   - 解释在正确答案标记之后
# 
# 解析策略：
#   1. 逐页处理（每页一个题目）
#   2. 使用状态机识别题目、选项、解释部分
#   3. 通过正则表达式提取结构化信息
#   4. 清理和验证提取的内容
# 
# 常见问题处理：
#   - 页眉页脚噪音：通过关键词过滤
#   - 不完整题目：检查题目是否以小写开头（可能是续行）
#   - 选项缺失：要求至少 2 个选项才保存题目
# 
# =============================================================================

def parse_surgery_pdf(pdf_path: str) -> List[Dict]:
    """
    解析 Surgery Qbank PDF (NBME 格式)。
    
    PDF 结构示例：
        Page 1:
        ■ Mark for Review
        1. A 25-year-old presents with RLQ pain...
        A) Order an abdominal ultrasound
        B) Start broad-spectrum IV antibiotics
        C) Schedule elective diagnostic laparoscopy
        D) Discharge home with oral analgesics
        Correct Answer: A. First-line test for suspected appendicitis...
        [解释文本继续...]
    
    解析流程：
        1. 加载 PDF 文件
        2. 逐页提取文本
        3. 检查是否有 "Correct Answer:" 标记（判断是否为有效题目页）
        4. 使用状态机识别题目、选项、解释部分
        5. 清理和验证提取的内容
        6. 构建题目对象并添加到列表
    
    Args:
        pdf_path: PDF 文件路径（字符串）
    
    Returns:
        List[Dict]: 题目列表，每个题目包含：
            - question: 题目文本
            - options: 选项字典 {"A": "选项A", "B": "选项B", ...}
            - correct_answer: 正确答案字母 ("A"-"E")
            - explanation: 解释文本
            - source: "Surgery Qbank"
            - source_file: PDF 文件名
            - page_number: 页码（从 1 开始）
    
    状态机说明：
        - in_question: 是否在收集题目文本
        - in_explanation: 是否在收集解释文本
        - 状态转换：
          题目开始标记 → in_question = True
          选项行 → in_question = False
          "Correct Answer:" → in_explanation = True
    
    注意事项：
        - 如果页面没有 "Correct Answer:" 标记，跳过该页
        - 如果题目以小写开头，可能是上一页的续行，跳过
        - 至少需要 2 个选项才保存题目
    """
    if not DEPS['pypdf']:
        print("Error: PyPDF not available")
        return []
    
    PyPDFLoader = DEPS['pypdf']
    loader = PyPDFLoader(pdf_path)
    
    try:
        docs = loader.load()
    except Exception as e:
        print(f"  Error loading PDF: {e}")
        return []
    
    questions = []
    
    # ========================================================================
    # 逐页处理
    # ========================================================================
    # enumerate(docs, start=1): 从 1 开始编号（人类可读的页码）
    for page_num, doc in enumerate(docs, start=1):
        text = doc.page_content  # 获取页面文本内容
        
        # ====================================================================
        # 步骤 1：检查页面是否包含有效题目
        # ====================================================================
        # 通过查找 "Correct Answer:" 标记来判断
        # 如果没有这个标记，说明这一页不是题目页（可能是封面、目录等）
        correct_match = re.search(r'Correct\s+Answer:\s*([A-J])\.?', text, re.IGNORECASE)
        if not correct_match:
            continue  # 跳过没有答案标记的页面
        
        # 提取正确答案字母（A-J，支持最多10个选项）
        correct_answer = correct_match.group(1).upper()
        
        # ====================================================================
        # 步骤 2：按行分割文本，准备逐行解析
        # ====================================================================
        lines = text.split('\n')
        
        # 初始化收集器
        question_lines = []      # 题目文本行
        options = {}              # 选项字典 {"A": "选项文本", ...}
        explanation_lines = []    # 解释文本行
        
        # 状态机标志
        in_question = False      # 是否正在收集题目文本
        in_explanation = False   # 是否正在收集解释文本
        
        # ====================================================================
        # 步骤 3：逐行解析，使用状态机识别不同部分
        # ====================================================================
        for line in lines:
            line_clean = line.strip()  # 去除首尾空白
            
            # ================================================================
            # 处理空行
            # ================================================================
            # 如果不在题目收集状态，跳过空行
            # 如果在题目收集状态，保留空行（可能是表格数据的一部分）
            if not line_clean:
                if in_question:
                    # 在题目状态中，保留空行（可能是表格分隔符）
                    question_lines.append('')
                continue
            
            # ================================================================
            # 过滤：跳过页眉页脚噪音
            # ================================================================
            # 这些是 PDF 中常见的导航元素，不是题目内容
            if any(x in line_clean for x in ['Exam Section', 'National Board', 'Score Report', 
                                              'Calculator', 'Previous', 'Next', 'https://']):
                continue
            
            # ================================================================
            # 识别：题目开始标记
            # ================================================================
            # Surgery PDF 中，题目通常以以下标记开始：
            #   - '■' (实心方块符号)
            #   - "'I" (可能是 PDF 提取错误，实际是题目开始标记)
            if '■' in line_clean or line_clean.startswith("'I"):
                in_question = True  # 进入题目收集状态
                
                # 清理题目开始标记
                line_clean = re.sub(r"^'I\s*", "", line_clean)  # 移除 "'I" 标记
                line_clean = re.sub(r'^■\s*Mark.*?Assessment\s*', '', line_clean)  # 移除 "■ Mark for Review Assessment" 等
                line_clean = re.sub(r'^\d+\.\s*', '', line_clean)  # 移除题号（如 "1. "）
                
                # 如果清理后还有内容，添加到题目文本
                if line_clean:
                    question_lines.append(line_clean)
                continue  # 处理下一行
            
            # ================================================================
            # 识别：选项行（改进版，支持多种格式，避免误识别实验数据）
            # ================================================================
            # 选项格式支持：
            #   - A) 选项文本
            #   - A. 选项文本
            #   - A 选项文本（较少见，但也要支持）
            # 
            # 但需要排除实验数据（如 "A 52-year-old" 或包含数字+单位的行）
            # 
            # 实验数据的特征：
            #   - 包含数字和单位（如 "10.2 g/dL", "100/mm3", "37.3°C"）
            #   - 包含冒号（如 "Hemoglobin:"）
            #   - 通常是表格格式
            # 
            # 选项的特征：
            #   - 以字母 A-J 开头，后跟右括号、点号或空格
            #   - 选项文本通常是完整的句子或短语
            #   - 不包含实验数据的典型格式
            
            # 支持多种选项格式：A) 或 A. 或 A （后跟空格）
            option_match = re.match(r'^([A-J])[)\.]\s*(.+)$', line_clean)
            if not option_match:
                # 尝试 A 选项文本格式（较少见）
                option_match = re.match(r'^([A-J])\s+(.+)$', line_clean)
            if option_match:
                letter = option_match.group(1)  # 选项字母
                opt_text = option_match.group(2).strip()  # 选项文本
                
                # 检查是否是实验数据（误匹配）- 更严格的判断
                # 实验数据通常包含：数字+单位、冒号、表格格式
                # 但选项也可能包含数字（如 "2.5 mg of medication"），所以判断要更谨慎
                is_lab_data = False
                
                # 检查1：包含明确的实验单位（更严格的模式）
                lab_units = r'(g/dL|mg/dL|mEq/L|U/L|mm3|/mm3|°C|°F|mmHg|/hpf|ng/mL|pg/cell|/L|/dL|/mL|IU/L)'
                if re.search(r'\d+\.?\d*\s*' + lab_units, opt_text, re.IGNORECASE):
                    # 进一步检查：如果选项文本很短（<15字符），且主要是数字+单位，才是实验数据
                    if len(opt_text.strip()) < 15:
                        is_lab_data = True
                
                # 检查2：包含冒号和数字，且格式像表格（如 "Hemoglobin: 10.2"）
                if ':' in opt_text and re.search(r'[A-Za-z]+\s*:\s*\d+', opt_text):
                    # 如果选项文本很短（<20字符），可能是实验数据
                    if len(opt_text.strip()) < 20:
                        is_lab_data = True
                
                # 检查3：太短且只包含数字和单位（<10字符）
                if len(opt_text.strip()) < 10 and re.search(r'^\d+\.?\d*\s*' + lab_units + r'$', opt_text, re.IGNORECASE):
                    is_lab_data = True
                
                if is_lab_data:
                    # 这是实验数据，不是选项，继续收集到题目中
                    if in_question:
                        question_lines.append(line_clean)
                    continue
                
                # 清理选项文本中的下划线（PDF 中的填空线）
                # 移除末尾的连续下划线（3 个或更多）
                opt_text = re.sub(r'[,\s]*_{3,}\s*$', '', opt_text)
                # 移除末尾的下划线 + 空格组合
                opt_text = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', opt_text)
                # 移除文本末尾的下划线模式
                opt_text = re.sub(r'[,\s]+_+\s*$', '', opt_text)
                # 移除文本中间的长下划线（可能是分隔符）
                opt_text = re.sub(r'_{4,}', '', opt_text)  # 移除 4 个或更多连续下划线
                
                opt_text = opt_text.strip()
                
                if opt_text:  # 只保存非空选项
                    options[letter] = opt_text
                in_question = False  # 遇到选项，题目部分结束
                continue  # 处理下一行
            
            # ================================================================
            # 识别：正确答案行（解释开始）
            # ================================================================
            # "Correct Answer: A" 这一行通常包含答案和解释的开头
            if 'Correct Answer:' in line_clean:
                in_explanation = True  # 进入解释收集状态
                
                # 提取 "Correct Answer: A" 之后的内容（解释的开头）
                # 正则说明：
                #   Correct\s+Answer:\s*[A-J]\.?\s*  - 匹配 "Correct Answer: A" 到 "Correct Answer: J"
                #   后面是解释文本的开始
                exp_start = re.sub(r'Correct\s+Answer:\s*[A-J]\.?\s*', '', line_clean)
                if exp_start.strip():
                    explanation_lines.append(exp_start.strip())
                continue  # 处理下一行
            
            # ================================================================
            # 收集：题目文本（在题目状态中）
            # ================================================================
            if in_question:
                line_clean = re.sub(r'^\d+\.\s*', '', line_clean)  # 移除可能的题号
                if line_clean:
                    question_lines.append(line_clean)
            
            # ================================================================
            # 特殊处理：在题目状态中，即使不在 in_question，也要收集实验数据
            # ================================================================
            # 实验数据可能在选项之前，需要被收集到题目文本中
            # 实验数据的特征：
            #   - 包含实验室检查关键词（如 "Laboratory studies", "Lab studies"）
            #   - 包含数字和单位（如 "10.2 g/dL", "100/mm3"）
            #   - 包含冒号和数值（如 "Hemoglobin: 10.2"）
            if not in_question and not in_explanation and len(options) == 0:
                # 检查是否是实验数据行
                is_lab_data_line = False
                
                # 检查是否包含实验室检查关键词
                lab_keywords = ['Laboratory studies', 'Lab studies', 'Laboratory:', 'Lab:', 
                               'Hematocrit', 'Hemoglobin', 'Leukocyte', 'Platelet', 'Serum',
                               'Na+', 'K+', 'Cl-', 'HCO3-', 'Creatinine', 'Glucose', 'Urea',
                               'Alkaline phosphatase', 'AST', 'ALT', 'Amylase', 'Lipase',
                               'Troponin', 'LDH', 'MCV', 'MCH', 'WBC', 'RBC']
                
                if any(keyword.lower() in line_clean.lower() for keyword in lab_keywords):
                    is_lab_data_line = True
                
                # 检查是否包含数字+单位的模式
                if re.search(r'\d+\.?\d*\s*(g/dL|mg/dL|mEq/L|U/L|mm3|/mm3|°C|°F|mmHg|/hpf|ng/mL|pg/cell|µm|kg/m2)', line_clean, re.IGNORECASE):
                    is_lab_data_line = True
                
                # 检查是否是表格格式（包含冒号和数值）
                if ':' in line_clean and re.search(r'\d+', line_clean):
                    # 格式如 "Hemoglobin: 10.2 g/dL" 或 "Na+ 137 mEq/L"
                    is_lab_data_line = True
                
                if is_lab_data_line:
                    # 这是实验数据，收集到题目文本中
                    question_lines.append(line_clean)
                    continue  # 跳过后续处理
            
            # ================================================================
            # 收集：解释文本（在解释状态中）
            # ================================================================
            if in_explanation:
                explanation_lines.append(line_clean)
        
        # ====================================================================
        # 步骤 4：合并和清理提取的内容
        # ====================================================================
        # 使用 clean_question_text() 保留换行符，以便保留表格数据格式
        # 用换行符连接题目行，而不是空格，这样可以保留表格结构
        question_text = clean_question_text('\n'.join(question_lines))
        
        # 格式化实验数据（使其更清晰易读）
        question_text = format_lab_data(question_text)
        
        # 解释文本使用 clean_explanation_text() 保留段落结构，便于阅读
        # 用换行符连接解释行，保留原始段落结构
        explanation = clean_explanation_text('\n'.join(explanation_lines))
        
        # ====================================================================
        # 步骤 5：验证题目完整性
        # ====================================================================
        # 检查题目是否以小写开头
        # 如果以小写开头，可能是上一页题目的续行，不是新题目
        if question_text and question_text[0].islower():
            continue  # 跳过不完整的题目
        
        # ====================================================================
        # 步骤 6：提取图片（如果有）
        # ====================================================================
        # 从当前页面提取图片
        image_paths = extract_images_from_page(pdf_path, page_num)
        
        # ====================================================================
        # 步骤 7：最终清理选项中的下划线
        # ====================================================================
        # 清理选项文本中的下划线（PDF 中的填空线）
        cleaned_options = {}
        for letter, opt_text in options.items():
            # 移除末尾的连续下划线（3 个或更多）
            cleaned_opt = re.sub(r'[,\s]*_{3,}\s*$', '', opt_text)
            # 移除末尾的下划线 + 空格组合
            cleaned_opt = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', cleaned_opt)
            # 移除文本末尾的下划线模式
            cleaned_opt = re.sub(r'[,\s]+_+\s*$', '', cleaned_opt)
            # 移除文本中间的长下划线（可能是分隔符）
            cleaned_opt = re.sub(r'_{4,}', '', cleaned_opt)  # 移除 4 个或更多连续下划线
            
            cleaned_opt = cleaned_opt.strip()
            
            if cleaned_opt:  # 只保留非空选项
                cleaned_options[letter] = cleaned_opt
        
        # ====================================================================
        # 步骤 8：保存有效题目
        # ====================================================================
        # 要求：至少要有题目文本和 2 个选项
        if question_text and len(cleaned_options) >= 2:
            question_obj = {
                "question": question_text,
                "options": cleaned_options,
                "correct_answer": correct_answer,
                "explanation": explanation,  # 可能为空（某些 PDF 没有解释）
                "source": "Surgery Qbank",
                "source_file": Path(pdf_path).name,  # 只保存文件名，不包含路径
                "page_number": page_num
            }
            
            # 如果有图片，添加到题目对象中
            if image_paths:
                question_obj["images"] = image_paths
            
            questions.append(question_obj)
    
    return questions


# =============================================================================
# Amboss PDF 解析器 (文本 PDF)
# =============================================================================
# 
# 功能：解析 Amboss Qbank PDF 文件
# 
# PDF 格式特征：
#   - 题目以数字开头：1. A 29-year-old man...
#   - 问题以 "?" 结尾
#   - Tip 部分：Tip: [提示文本]
#   - 选项格式：选项文本 百分比%
#   - 正确答案：百分比最高的选项
#   - 多个题目可能在同一页
# 
# 解析策略：
#   1. 将所有页面合并为完整文本
#   2. 按题号分割（通过正则匹配题号模式）
#   3. 对每个题目部分：
#      - 提取问题文本（到 "?" 为止）
#      - 提取 Tip（如果存在）
#      - 提取选项（通过百分比模式匹配）
#      - 确定正确答案（最高百分比）
# 
# 已知问题：
#   1. Tip 提取可能不完整：
#      - 正则表达式 r'Tip:\s*([^%]+?)(?=\s+[A-Z][a-z].*?\d+%|$)'
#        依赖于选项格式（大写字母开头 + 百分比）
#      - 如果 Tip 和选项之间没有这种格式，可能匹配失败
#      - 如果 Tip 中包含 "%" 符号，也会提前终止匹配
#   
#   2. Explanation 为空：
#      - Amboss PDF 格式中，解释通常不在题目页面
#      - 可能在其他页面或单独的文件中
#      - 当前实现没有提取解释的逻辑
# 
# 改进建议：
#   - 改进 Tip 提取：使用更宽松的模式，或逐行查找 "Tip:" 标记
#   - 添加解释提取：如果 PDF 中有解释部分，添加提取逻辑
# 
# =============================================================================

def parse_amboss_pdf(pdf_path: str) -> List[Dict]:
    """
    解析 Amboss Qbank PDF。
    
    PDF 结构示例：
        1. A 29-year-old man presents with...
        Tip: Consider the mechanism of action...
        Administration of antivenom 69%
        Supportive care only 12%
        Observation 10%
        Surgical intervention 9%
    
    解析流程：
        1. 加载 PDF 并合并所有页面文本
        2. 为每页添加页码标记（用于追踪题目来源）
        3. 按题号分割文本（通过正则匹配题号模式）
        4. 对每个题目部分：
           a. 提取问题文本（数字开头到 "?" 结尾）
           b. 提取 Tip（如果存在）
           c. 提取选项（通过百分比模式）
           d. 确定正确答案（最高百分比）
        5. 构建题目对象
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        List[Dict]: 题目列表，每个题目包含：
            - question: 题目文本
            - options: 选项字典 {"A": "选项A", ...}
            - correct_answer: 正确答案字母（百分比最高的选项）
            - tip: 提示文本（可能为空）
            - explanation: 解释文本（通常为空，Amboss 格式中没有）
            - source: "Amboss Qbank"
            - source_file: PDF 文件名
            - page_number: 页码
    
    注意事项：
        - 选项通过百分比确定，最高百分比的选项是正确答案
        - Tip 提取可能不完整（见上方注释）
        - Explanation 通常为空（Amboss PDF 格式限制）
    """
    if not DEPS['pypdf']:
        print("Error: PyPDF not available")
        return []
    
    PyPDFLoader = DEPS['pypdf']
    loader = PyPDFLoader(pdf_path)
    
    try:
        docs = loader.load()
    except Exception as e:
        print(f"  Error loading PDF: {e}")
        return []
    
    # ========================================================================
    # 步骤 1：合并所有页面文本，并添加页码标记
    # ========================================================================
    # 为什么添加页码标记？
    #   - Amboss PDF 中，多个题目可能在同一页
    #   - 需要追踪每个题目来自哪一页
    #   - 标记格式：[PAGE:1], [PAGE:2], ...
    page_texts = []
    for page_num, doc in enumerate(docs, start=1):
        page_texts.append(f"[PAGE:{page_num}]\n{doc.page_content}")
    
    # 合并所有页面文本
    full_text = "\n".join(page_texts)
    questions = []
    
    # ========================================================================
    # 步骤 2：按题号分割文本（改进版）
    # ========================================================================
    # 改进策略：使用更通用的模式来识别题目边界
    # 
    # 题目开头的常见模式：
    #   - "1. A 29-year-old man..."
    #   - "2. A 22-year-old woman..."
    #   - "3. A 7-year-old boy..."
    #   - "4. A 66-year-old man..."
    #   - "5. A 72-year-old man..."（后面可能格式不同）
    #   - 等等
    # 
    # 新的正则表达式：r'(?=\d+\.\s+A\s+\d+-year-old\s+(?:man|woman|boy|girl|patient|person))'
    #   - (?=...) 是正向前瞻（lookahead），匹配但不消耗字符
    #   - \d+\.\s+ 匹配题号（如 "1. "）
    #   - A\s+\d+-year-old\s+ 匹配年龄模式
    #   - (?:man|woman|boy|girl|patient|person) 匹配常见的人物描述
    # 
    # 如果上面的模式匹配不到，尝试更宽松的模式：
    #   - r'(?=\d+\.\s+[A-Z][a-z])' - 题号后跟大写字母开头的单词
    #   - 但需要排除一些误匹配（如 "A. " 开头的选项）
    # 
    # 改进：使用更灵活的分割策略
    #   1. 首先尝试匹配年龄模式（最常见）
    #   2. 如果失败，尝试匹配题号 + 大写字母开头的句子（更宽松）
    #   3. 确保分割后的每个部分都以题号开头
    parts = re.split(r'(?=\d+\.\s+A\s+\d+-year-old\s+(?:man|woman|boy|girl|patient|person))', full_text, flags=re.IGNORECASE)
    
    # 如果分割后只有一个部分，尝试更宽松的模式
    if len(parts) <= 1:
        # 更宽松的模式：题号后跟大写字母开头的单词（至少3个字符）
        # 排除 "A. " 这种选项格式
        parts = re.split(r'(?=\d+\.\s+[A-Z][a-z]{2,})', full_text)
    
    # 进一步验证：确保每个部分都以题号开头
    # 过滤掉不以题号开头的部分（可能是分割错误）
    valid_parts = []
    for part in parts:
        if part.strip() and re.match(r'\d+\.\s+', part.strip()):
            valid_parts.append(part)
    parts = valid_parts
    
    # ========================================================================
    # 步骤 3：处理每个题目部分
    # ========================================================================
    for part in parts:
        if not part.strip():  # 跳过空部分
            continue
        
        # 验证是否以题号开头
        num_match = re.match(r'(\d+)\.\s+', part)
        if not num_match:
            continue
        
        # ====================================================================
        # 提取页码（从之前添加的标记中）
        # ====================================================================
        page_match = re.search(r'\[PAGE:(\d+)\]', part)
        page_num = int(page_match.group(1)) if page_match else 1
        
        try:
            # ================================================================
            # 步骤 3.1：提取问题文本（改进版）
            # ================================================================
            # 改进策略：
            #   1. 首先尝试匹配到问号（最常见的情况）
            #   2. 如果没有问号，尝试匹配到选项开始（通过百分比模式）
            #   3. 如果都没有，匹配到 "Tip:" 或文本结束
            question_text = ""
            
            # 方法1：尝试匹配到问号，然后继续提取 Tip（如果存在）
            # Tip 内容应该保留在题目文本中
            q_match = re.search(r'^(\d+\.\s+.*?\?)', part, re.DOTALL)
            if q_match:
                # 提取到问号的内容
                question_base = clean_question_text(q_match.group(1))
                question_base = re.sub(r'^\d+\.\s+', '', question_base)  # 移除题号
                
                # 格式化实验数据
                question_base = format_lab_data(question_base)
                
                # 检查问号之后是否有 Tip
                question_end_pos = q_match.end()
                remaining_text = part[question_end_pos:].strip()
                
                # 查找 Tip 标记
                tip_pattern = re.compile(r'Tip\s*:', re.IGNORECASE)
                tip_match = tip_pattern.search(remaining_text)
                if tip_match:
                    # 找到 Tip，提取 Tip 内容直到选项开始
                    tip_start_pos = tip_match.end()
                    tip_text = remaining_text[tip_start_pos:].strip()
                    
                    # 查找选项开始的位置（通过百分比模式）
                    option_start_match = re.search(r'\s+([A-Z][a-z][^%]{3,}?)\s+\d{1,2}%', tip_text)
                    if option_start_match:
                        # 提取完整的 Tip 内容
                        tip_content = tip_text[:option_start_match.start()].strip()
                        # 将 Tip 添加到题目文本中
                        question_text = question_base + "\n\nTip: " + tip_content
                    else:
                        # 没有找到选项，提取 Tip 内容（限制长度）
                        tip_content = tip_text[:2000].strip()  # 增加到 2000 字符
                        question_text = question_base + "\n\nTip: " + tip_content
                else:
                    # 没有 Tip，只使用问题文本
                    question_text = question_base
            else:
                # 方法2：如果没有问号，尝试匹配到选项开始
                    # 选项通常以大写字母开头 + 百分比结尾
                    # Tip 内容应该保留在题目文本中
                    option_start_match = re.search(r'\s+([A-Z][a-z][^%]{3,}?)\s+\d{1,2}%', part)
                    if option_start_match:
                        # 提取到选项之前的内容（包括 Tip）
                        question_text = part[:option_start_match.start()].strip()
                        question_text = re.sub(r'^\d+\.\s+', '', question_text)  # 移除题号
                        question_text = clean_question_text(question_text)
                        # 格式化实验数据
                        question_text = format_lab_data(question_text)
                    else:
                        # 方法3：匹配到选项开始或文本结束
                        # Tip 内容应该保留在题目文本中
                        # 如果没有找到选项，提取到文本结束（但限制长度）
                        question_text = part[:2000].strip()  # 增加长度限制，确保包含 Tip
                        question_text = re.sub(r'^\d+\.\s+', '', question_text)  # 移除题号
                        question_text = clean_question_text(question_text)
                        # 格式化实验数据
                        question_text = format_lab_data(question_text)
            
            # 如果问题文本为空或太短，跳过
            if not question_text or len(question_text.strip()) < 10:
                continue
            
            # ================================================================
            # 步骤 3.2：提取 Tip（如果存在），但保留在题干中
            # ================================================================
            # Tip 内容应该在题干中（格式：\n\nTip: ...）
            # 提取 tip 用于单独字段，但不要从 question_text 中移除
            tip = ""
            tip_pattern = re.compile(r'\n\nTip:\s*(.+?)(?=\n\n|$)', re.DOTALL | re.IGNORECASE)
            tip_match = tip_pattern.search(question_text)
            if tip_match:
                tip = tip_match.group(1).strip()
                # 注意：不从这里移除 Tip，保留在 question_text 中
                # question_text 已经包含了 Tip，所以不需要移除
            
            # ================================================================
            # 步骤 3.3：提取选项（通过百分比模式，改进版，支持更多格式）
            # ================================================================
            # 改进策略：
            #   1. 使用更灵活的正则表达式，支持更多选项格式
            #   2. 支持以大写字母开头的选项（常见）
            #   3. 支持以数字开头的选项（如 "2.5 mg"）
            #   4. 限制选项提取范围：只从题目结束到解释开始之间提取
            #   5. 过滤明显不是选项的文本（如解释文本）
            # 
            # 正则说明：
            #   模式1：r'([A-Z][^%]{2,200}?)\s+(\d{1,2})%' - 大写字母开头的选项
            #   模式2：r'(\d+[^%]{1,200}?)\s+(\d{1,2})%' - 数字开头的选项（如 "2.5 mg 69%"）
            # 
            # 为什么支持数字开头？
            #   - 有些选项可能以数字开头（如 "2.5 mg of medication"）
            #   - 但需要确保不是实验数据（通过后续过滤）
            # 
            # 限制选项提取范围：
            #   - 只从题目结束到解释开始之间提取选项
            #   - 如果找到解释标记（如 "Explanation:", "Answer:"），停止提取
            option_patterns = [
                r'([A-Z][^%]{2,200}?)\s+(\d{1,2})%',  # 大写字母开头的选项（更宽松）
                r'(\d+[^%]{1,200}?)\s+(\d{1,2})%',    # 数字开头的选项（如 "2.5 mg 69%"）
            ]
            
            # 确定选项提取的范围
            # 从题目文本结束到解释开始之间提取选项
            # 改进：如果找不到题目结束位置，从题号后开始搜索
            option_extraction_start = 0
            if question_text:
                # 找到题目文本在 part 中的位置（保留 Tip 部分，因为 Tip 应该在题干中）
                # 尝试匹配题目文本的前50个字符（包含 Tip）
                if len(question_text) > 20:
                    question_end_match = re.search(re.escape(question_text[:50]), part, re.IGNORECASE)
                    if question_end_match:
                        option_extraction_start = question_end_match.end()
                    else:
                        # 如果匹配不到，尝试匹配前30个字符
                        question_end_match = re.search(re.escape(question_text[:30]), part, re.IGNORECASE)
                        if question_end_match:
                            option_extraction_start = question_end_match.end()
                        else:
                            # 如果还是匹配不到，尝试查找问号或 Tip 结束位置
                            # 查找问号位置
                            q_mark_pos = part.find('?')
                            if q_mark_pos > 0:
                                # 查找问号后的 Tip 结束位置
                                tip_end_pattern = re.search(r'\?\s*Tip:\s*.*?(?=\s+[A-Z][a-z][^%]{3,}?\s+\d{1,2}%|$)', part[q_mark_pos:], re.DOTALL | re.IGNORECASE)
                                if tip_end_pattern:
                                    option_extraction_start = q_mark_pos + tip_end_pattern.end()
                                else:
                                    option_extraction_start = q_mark_pos + 1
            
            # 如果还是找不到，从题号后开始（至少跳过题号部分）
            if option_extraction_start == 0:
                num_match = re.match(r'\d+\.\s+', part)
                if num_match:
                    option_extraction_start = num_match.end()
            
            # 查找解释开始的位置（如果存在）
            explanation_markers = [
                r'Explanation\s*:',
                r'Answer\s*:',
                r'Rationale\s*:',
                r'Note\s*:',
                r'Correct\s+Answer\s*:',
            ]
            option_extraction_end = len(part)
            for marker in explanation_markers:
                marker_match = re.search(marker, part[option_extraction_start:], re.IGNORECASE)
                if marker_match:
                    # 找到解释标记，限制选项提取范围
                    option_extraction_end = option_extraction_start + marker_match.start()
                    break
            
            # 只在选项提取范围内搜索选项
            # 如果范围太小（小于50字符），可能题目文本定位有问题，扩大搜索范围
            if option_extraction_end - option_extraction_start < 50:
                # 扩大搜索范围到整个 part 的前80%
                option_extraction_end = int(len(part) * 0.8)
            
            option_text_range = part[option_extraction_start:option_extraction_end]
            # 使用多个模式匹配选项
            all_matches = []
            for pattern in option_patterns:
                matches = re.findall(pattern, option_text_range)
                all_matches.extend(matches)
            
            # 去重：如果同一个百分比出现多次，只保留第一个匹配
            seen_percent_pos = {}  # {percent: position} 用于去重
            unique_matches = []
            for match in all_matches:
                opt_text, percent = match
                # 检查是否已经见过这个百分比（在相似位置）
                if percent not in seen_percent_pos:
                    unique_matches.append(match)
                    # 记录这个百分比在文本中的位置（用于去重）
                    pos = option_text_range.find(opt_text + ' ' + percent + '%')
                    if pos >= 0:
                        seen_percent_pos[percent] = pos
                else:
                    # 如果百分比已存在，检查位置是否接近（可能是重复匹配）
                    pos = option_text_range.find(opt_text + ' ' + percent + '%')
                    if pos >= 0 and abs(pos - seen_percent_pos[percent]) > 50:
                        # 位置相差较大，可能是不同的选项，保留
                        unique_matches.append(match)
            
            all_matches = unique_matches
            
            # 初始化选项收集器
            options = {}
            correct_answer = ""
            max_percent = 0
            seen = set()  # 用于去重（避免相同百分比重复）
            
            # 处理每个匹配的选项
            for opt_text, percent in all_matches:
                if len(options) >= 10:  # 最多 10 个选项（A-J）
                    break
                if percent in seen:  # 跳过重复的百分比
                    continue
                
                opt_clean = clean_text(opt_text)
                
                # 再次清理页码标记（确保完全移除）
                opt_clean = re.sub(r'\[PAGE:\d+\]', '', opt_clean).strip()
                
                # 清理选项文本中的下划线（PDF 中的填空线）
                # 移除末尾的连续下划线（3 个或更多）
                opt_clean = re.sub(r'[,\s]*_{3,}\s*$', '', opt_clean)
                # 移除末尾的下划线 + 空格组合
                opt_clean = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', opt_clean)
                # 移除文本末尾的下划线模式
                opt_clean = re.sub(r'[,\s]+_+\s*$', '', opt_clean)
                # 移除文本中间的长下划线（可能是分隔符）
                opt_clean = re.sub(r'_{4,}', '', opt_clean)  # 移除 4 个或更多连续下划线
                
                opt_clean = opt_clean.strip()
                
                # 过滤太短的选项（可能是误匹配）
                # 放宽限制：从 3 字符降到 2 字符，避免漏掉极短选项
                if len(opt_clean) < 2:
                    continue
                
                # 检查是否是实验数据（Amboss PDF 中也可能有实验数据混入选项）
                # 如果选项只包含数字+单位，且很短，可能是实验数据
                lab_units_pattern = r'(g/dL|mg/dL|mEq/L|U/L|mm3|/mm3|°C|°F|mmHg|/hpf|ng/mL|pg/cell|/L|/dL|/mL|IU/L)'
                if len(opt_clean) < 15 and re.search(r'^\d+\.?\d*\s*' + lab_units_pattern + r'$', opt_clean, re.IGNORECASE):
                    continue  # 跳过实验数据
                
                # 大幅减少过滤条件，只保留最明显的非选项标记
                # 只过滤明显是解释文本的内容
                exclusion_keywords = [
                    'Incorrect Answers', 'Explanation:', 'Answer:', 'Rationale:',
                    'This option is incorrect', 'This choice is incorrect', 'Tip:'
                ]
                # 只有当选项包含这些关键词时才过滤
                if any(x in opt_clean for x in exclusion_keywords):
                    continue
                
                # 放宽句子过滤：只过滤包含5个或更多句子的选项（可能是解释文本）
                sentence_endings = opt_clean.count('.') + opt_clean.count('!') + opt_clean.count('?')
                if sentence_endings > 4:  # 进一步放宽到4个句子
                    continue
                
                # 放宽长度限制：从300字符增加到500字符（支持更长的选项）
                if len(opt_clean) > 500:
                    continue
                
                # 移除所有解释性短语开头的过滤（这些可能是有效选项）
                # 不再过滤以 "This patient" 等开头的选项
                
                # 分配选项字母（A, B, C, ..., J）
                # chr(65) = 'A', chr(66) = 'B', ..., chr(74) = 'J'
                letter = chr(65 + len(options))
                options[letter] = opt_clean
                seen.add(percent)
                
                # 更新正确答案（百分比最高的选项）
                if int(percent) > max_percent:
                    max_percent = int(percent)
                    correct_answer = letter
            
            # ================================================================
            # 步骤 3.4：尝试提取 Explanation（如果有），并按选项拆分
            # ================================================================
            # Amboss PDF 中，解释可能在选项之后
            # 需要为每个选项提取对应的解释
            explanation = ""  # 保留整体解释作为后备
            explanation_raw = ""  # 原始解释文本（用于按选项拆分）
            option_explanations = {}  # 每个选项对应的解释字典（初始为空）
            
            # 找到最后一个选项的位置
            if all_matches:
                # 找到最后一个匹配的选项在原文中的位置
                last_option_match = None
                last_match_pos = -1
                # 使用所有模式查找最后一个选项
                for pattern in option_patterns:
                    for match in re.finditer(pattern, part):
                        if match.end() > last_match_pos:
                            last_match_pos = match.end()
                            last_option_match = match
                
                if last_option_match:
                    # 从最后一个选项之后提取可能的解释
                    explanation_start = last_option_match.end()
                    explanation_text = part[explanation_start:].strip()
                    
                    # 移除页码标记
                    explanation_text = re.sub(r'\[PAGE:\d+\]', '', explanation_text)
                    
                    # 检查是否有明显的解释标记（如 "Explanation:", "Answer:", 等）
                    # 使用正则表达式匹配，不区分大小写
                    explanation_markers = [
                        r'Explanation\s*:',
                        r'Answer\s*:',
                        r'Rationale\s*:',
                        r'Note\s*:',
                        r'Correct\s+Answer\s*:',
                    ]
                    
                    # 查找第一个匹配的标记
                    marker_match = None
                    marker_pos = -1
                    for marker in explanation_markers:
                        match = re.search(marker, explanation_text, re.IGNORECASE)
                        if match and (marker_pos == -1 or match.start() < marker_pos):
                            marker_match = match
                            marker_pos = match.start()
                    
                    if marker_match:
                        # 找到标记，提取标记之后的内容
                        explanation_text = explanation_text[marker_match.end():].strip()
                    
                    # 清理解释文本（保留段落结构）
                    if explanation_text:
                        # 查找可能的结束位置（下一题的开始）
                        next_question_match = re.search(r'\d+\.\s+A\s+\d+-year-old', explanation_text, re.IGNORECASE)
                        if next_question_match:
                            # 如果找到下一题，只提取到下一题之前
                            explanation_text = explanation_text[:next_question_match.start()].strip()
                        
                        # 增加长度限制到 5000 字符，确保提取完整的 explanation
                        if len(explanation_text) > 5000:
                            explanation_text = explanation_text[:5000].strip()
                        
                        # 移除可能的下一题标记
                        explanation_text = re.sub(r'\d+\.\s+A\s+\d+-year-old.*', '', explanation_text, flags=re.DOTALL)
                        
                        # 清理文本
                        explanation = clean_explanation_text(explanation_text)
                        
                        # 如果太短，可能是噪音
                        if len(explanation.strip()) < 10:
                            explanation = ""
                        
                        # 注意：这里还不能拆分解释，因为 cleaned_options 还没有准备好
                        # 将在步骤 3.7 中处理
            
            # ================================================================
            # 步骤 3.5：提取图片（如果有）
            # ================================================================
            # 从当前页面提取图片
            image_paths = extract_images_from_page(pdf_path, page_num)
            
            # ================================================================
            # 步骤 3.6：最终清理所有字段，移除页码标记，格式化实验数据
            # ================================================================
            # 确保所有字段都没有页码标记
            question_text = re.sub(r'\[PAGE:\d+\]', '', question_text).strip()
            
            # 格式化实验数据（使其更清晰易读）
            question_text = format_lab_data(question_text)
            
            tip = re.sub(r'\[PAGE:\d+\]', '', tip).strip() if tip else ""
            explanation = re.sub(r'\[PAGE:\d+\]', '', explanation).strip() if explanation else ""
            
            # 清理选项中的页码标记和下划线
            cleaned_options = {}
            for letter, opt_text in options.items():
                cleaned_opt = re.sub(r'\[PAGE:\d+\]', '', opt_text).strip()
                
                # 清理下划线（PDF 中的填空线）
                # 移除末尾的连续下划线（3 个或更多）
                cleaned_opt = re.sub(r'[,\s]*_{3,}\s*$', '', cleaned_opt)
                # 移除末尾的下划线 + 空格组合
                cleaned_opt = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', cleaned_opt)
                # 移除文本末尾的下划线模式
                cleaned_opt = re.sub(r'[,\s]+_+\s*$', '', cleaned_opt)
                # 移除文本中间的长下划线（可能是分隔符）
                cleaned_opt = re.sub(r'_{4,}', '', cleaned_opt)  # 移除 4 个或更多连续下划线
                
                cleaned_opt = cleaned_opt.strip()
                
                if cleaned_opt:  # 只保留非空选项
                    cleaned_options[letter] = cleaned_opt
            
            # ================================================================
            # 步骤 3.7：按选项拆分解释
            # ================================================================
            # 如果有解释和选项，尝试按选项拆分解释
            if (explanation or explanation_raw) and cleaned_options and correct_answer:
                # 使用原始解释文本进行拆分（保留更多格式信息）
                raw_text_to_parse = explanation_raw if explanation_raw else explanation
                option_explanations = parse_option_explanations(
                    raw_text_to_parse,
                    cleaned_options, 
                    correct_answer
                )
            else:
                option_explanations = {}
            
            # ================================================================
            # 步骤 3.8：保存题目
            # ================================================================
            # 要求：至少要有题目文本和 1 个选项（放宽条件，避免漏题）
            # 确保 tip 包含在 question_text 中
            if tip and tip not in question_text:
                # 如果 tip 不在题干中，添加到题干末尾
                question_text = question_text + "\n\nTip: " + tip
            
            if question_text and len(cleaned_options) >= 1:
                question_obj = {
                    "question": question_text,  # question_text 已经包含 tip
                    "options": cleaned_options,
                    "correct_answer": correct_answer if correct_answer else "A",  # 默认 A
                    "tip": tip,  # 保留 tip 字段用于 UI 显示（但主要应该在 question 中）
                    "explanation": explanation,  # 整体解释（作为后备）
                    "option_explanations": option_explanations if option_explanations else None,  # 每个选项的解释
                    "source": "Amboss Qbank",
                    "source_file": Path(pdf_path).name,
                    "page_number": page_num
                }
                
                # 如果有图片，添加到题目对象中
                if image_paths:
                    question_obj["images"] = image_paths
                
                questions.append(question_obj)
        except Exception:
            # 如果解析某个题目时出错，跳过它，继续处理下一个
            continue
    
    return questions


# =============================================================================
# OCR PDF 解析器 (扫描 PDF)
# =============================================================================
# 
# 功能：使用 OCR（光学字符识别）解析扫描的 PDF 文件
# 
# 使用场景：
#   - PDF 是扫描的图片，没有可提取的文本
#   - 需要通过 OCR 将图片中的文字转换为文本
#   - 然后使用与文本 PDF 类似的解析逻辑
# 
# 技术流程：
#   1. 使用 PyMuPDF 将 PDF 页面渲染为图片
#   2. 使用 Tesseract OCR 识别图片中的文字
#   3. 对 OCR 结果使用正则表达式提取题目信息
# 
# 依赖要求：
#   - PyMuPDF: 将 PDF 页面渲染为图片
#   - Tesseract OCR: 识别图片中的文字
#   - PIL/Pillow: 图像处理
# 
# OCR 准确性：
#   - OCR 结果可能包含识别错误
#   - 正则表达式需要更宽松，容忍 OCR 错误
#   - 某些格式可能无法准确识别
# 
# 性能考虑：
#   - OCR 处理较慢（每页可能需要几秒）
#   - 对于大文件，处理时间可能很长
#   - 建议只对扫描 PDF 使用此方法
# 
# =============================================================================

def parse_scanned_pdf(pdf_path: str, poppler_path: Optional[str] = None) -> List[Dict]:
    """
    使用 OCR 解析扫描的 PDF 文件。
    
    处理流程：
        1. 打开 PDF 文件（使用 PyMuPDF）
        2. 逐页处理：
           a. 将页面渲染为高分辨率图片（2x 缩放）
           b. 使用 Tesseract OCR 识别图片中的文字
           c. 对 OCR 文本使用正则表达式提取题目信息
        3. 返回题目列表
    
    Args:
        pdf_path: PDF 文件路径
        poppler_path: Poppler bin 目录路径（Windows，可选）
                     注意：当前实现使用 PyMuPDF，不需要 Poppler
                     此参数保留是为了兼容性
    
    Returns:
        List[Dict]: 题目列表，格式与 parse_surgery_pdf 相同
    
    注意事项：
        - OCR 识别可能不准确，特别是：
          * 手写文字
          * 低质量扫描
          * 复杂布局
          * 特殊符号
        - 正则表达式需要容忍 OCR 错误（如 "0" vs "O"）
        - 处理速度较慢，大文件可能需要很长时间
    """
    if not DEPS['pymupdf']:
        print("Error: PyMuPDF not available. Run: pip install pymupdf")
        return []
    
    if not DEPS['tesseract']:
        print("Error: Tesseract not available. Run: pip install pytesseract")
        return []
    
    fitz = DEPS['pymupdf']
    pytesseract = DEPS['tesseract']
    Image = DEPS['pil']
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Error opening PDF: {e}")
        return []
    
    print(f"  Processing {len(doc)} pages with OCR...")
    
    questions = []
    
    # ========================================================================
    # 逐页处理（OCR 处理）
    # ========================================================================
    for i, page in enumerate(doc):
        # 每 10 页显示一次进度（OCR 较慢，需要进度提示）
        if i % 10 == 0:
            print(f"    Page {i+1}/{len(doc)}...")
        
        try:
            # ================================================================
            # 步骤 1：将 PDF 页面渲染为图片
            # ================================================================
            # Matrix(2, 2): 2x 缩放因子
            # 为什么使用 2x 缩放？
            #   - 更高的分辨率可以提高 OCR 准确性
            #   - 但会增加处理时间和内存使用
            #   - 2x 是准确性和性能的平衡
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)  # 渲染为像素图
            
            # 转换为 PIL Image 对象（Tesseract 需要）
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # ================================================================
            # 步骤 2：OCR 识别图片中的文字
            # ================================================================
            # Tesseract OCR 将图片转换为文本
            # 注意：OCR 结果可能包含识别错误
            text = pytesseract.image_to_string(img)
            
            # ================================================================
            # 步骤 3：从 OCR 文本中提取题目信息
            # ================================================================
            # 使用与文本 PDF 类似的逻辑，但正则表达式更宽松（容忍 OCR 错误）
            
            # 检查是否有正确答案标记
            # 正则说明：r'Correct\s+Answer[:\s]*([A-J])'
            #   - Correct\s+Answer 匹配 "Correct Answer" 或 "Correct  Answer"（容忍多余空格）
            #   - [:\s]* 匹配 ":" 或空白字符（OCR 可能识别错误）
            #   - ([A-J]) 捕获答案字母（支持最多10个选项）
            correct_match = re.search(r'Correct\s+Answer[:\s]*([A-J])', text, re.IGNORECASE)
            if not correct_match:
                continue  # 没有找到答案标记，跳过该页
            
            correct_answer = correct_match.group(1).upper()
            
            # 提取问题文本
            # 正则说明：r'(\d+)[.\s]+([A-Z].*?\?)'
            #   - (\d+) 题号
            #   - [.\s]+ 点号或空白（OCR 可能识别错误）
            #   - ([A-Z].*?\?) 问题文本（以大写字母开头，到问号结束）
            q_match = re.search(r'(\d+)[.\s]+([A-Z].*?\?)', text, re.DOTALL)
            if not q_match:
                continue  # 没有找到问题，跳过
            
            # 使用 clean_question_text() 保留换行符，以便保留表格数据格式
            question_text = clean_question_text(q_match.group(2))
            
            # 提取选项（使用共享函数）
            options = extract_options(text)
            
            # 提取解释（保留段落结构）
            # 正则说明：r'Correct\s+Answer[:\s]*[A-J][.\s]*(.*?)(?:Incorrect|$)'
            #   - Correct\s+Answer[:\s]*[A-J][.\s]* 匹配答案行（支持 A-J）
            #   - (.*?) 捕获解释文本（非贪婪）
            #   - (?:Incorrect|$) 匹配 "Incorrect" 或行尾（解释结束标记）
            exp_match = re.search(r'Correct\s+Answer[:\s]*[A-J][.\s]*(.*?)(?:Incorrect|$)', 
                                  text, re.DOTALL | re.IGNORECASE)
            explanation = clean_explanation_text(exp_match.group(1)) if exp_match else ""
            
            # ================================================================
            # 步骤 4：提取图片（如果有）
            # ================================================================
            # 从当前页面提取嵌入的图片
            image_paths = extract_images_from_page(pdf_path, i + 1)
            
            # ================================================================
            # 步骤 5：清理选项中的下划线
            # ================================================================
            # 清理选项文本中的下划线（PDF 中的填空线）
            cleaned_options = {}
            for letter, opt_text in options.items():
                # 移除末尾的连续下划线（3 个或更多）
                cleaned_opt = re.sub(r'[,\s]*_{3,}\s*$', '', opt_text)
                # 移除末尾的下划线 + 空格组合
                cleaned_opt = re.sub(r'[,\s]*_\s+_{1,}\s*$', '', cleaned_opt)
                # 移除文本末尾的下划线模式
                cleaned_opt = re.sub(r'[,\s]+_+\s*$', '', cleaned_opt)
                # 移除文本中间的长下划线（可能是分隔符）
                cleaned_opt = re.sub(r'_{4,}', '', cleaned_opt)  # 移除 4 个或更多连续下划线
                
                cleaned_opt = cleaned_opt.strip()
                
                if cleaned_opt:  # 只保留非空选项
                    cleaned_options[letter] = cleaned_opt
            
            # ================================================================
            # 步骤 6：验证和保存题目
            # ================================================================
            if question_text and len(cleaned_options) >= 2:
                # 确保答案在选项中
                # OCR 可能识别错误，导致答案不在选项中
                if correct_answer not in cleaned_options:
                    # 如果答案不在选项中，使用第一个选项作为默认值
                    correct_answer = sorted(cleaned_options.keys())[0]
                
                question_obj = {
                    "question": question_text,
                    "options": cleaned_options,
                    "correct_answer": correct_answer,
                    "explanation": explanation,  # 可能为空（OCR 识别失败）
                    "source": "Surgery Qbank (OCR)",
                    "source_file": Path(pdf_path).name,
                    "page_number": i + 1
                }
                
                # 如果有图片，添加到题目对象中
                if image_paths:
                    question_obj["images"] = image_paths
                
                questions.append(question_obj)
        except Exception as e:
            # 如果某页 OCR 失败，跳过它，继续处理下一页
            # 不打印错误，避免输出过多信息
            continue
    
    doc.close()  # 关闭 PDF 文件
    return questions


# =============================================================================
# 自动检测 PDF 类型
# =============================================================================
# 
# 功能：自动检测 PDF 文件的类型，以便使用正确的解析器
# 
# 检测策略（按优先级）：
#   1. 文件名判断（快速，但不一定准确）
#   2. 文本内容分析（更准确）
#   3. 文本长度判断（区分文本 PDF 和扫描 PDF）
# 
# 检测逻辑：
#   - 如果文件名包含 "amboss" → 'amboss'
#   - 如果文件名包含 "surgery" → 进一步检查
#   - 如果文本长度 < 100 字符 → 'scanned'（扫描 PDF）
#   - 如果包含 "AMBOSS" 标记 → 'amboss'
#   - 如果包含 "Correct Answer:" 或 "NBME" → 'surgery'
#   - 默认 → 'surgery'
# 
# 为什么需要检测？
#   - 不同 PDF 格式需要不同的解析逻辑
#   - 用户可能不知道 PDF 类型
#   - 自动检测提高用户体验
# 
# =============================================================================

def detect_pdf_type(pdf_path: str) -> str:
    """
    自动检测 PDF 文件的类型。
    
    检测流程：
        1. 首先根据文件名快速判断（如果文件名包含关键词）
        2. 如果无法确定，加载 PDF 并分析内容
        3. 根据文本长度和特征标记判断类型
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        str: PDF 类型
            - 'surgery': Surgery Qbank 文本 PDF
            - 'amboss': Amboss 文本 PDF
            - 'scanned': 扫描 PDF（需要 OCR）
            - 'unknown': 无法确定类型
    
    检测规则：
        1. 文件名包含 "amboss" → 'amboss'
        2. 文件名包含 "surgery" → 检查文本长度
           - 文本 < 100 字符 → 'scanned'
           - 否则 → 'surgery'
        3. 文本内容包含 "AMBOSS" → 'amboss'
        4. 文本内容包含 "Correct Answer:" 或 "NBME" → 'surgery'
        5. 文本长度 < 100 字符 → 'scanned'
        6. 默认 → 'surgery'
    """
    # ========================================================================
    # 步骤 1：根据文件名快速判断（最快的方法）
    # ========================================================================
    filename = Path(pdf_path).name.lower()  # 转换为小写以便比较
    
    # 如果文件名包含 "amboss"，很可能是 Amboss PDF
    if 'amboss' in filename:
        return 'amboss'
    
    # 如果文件名包含 "surgery"，需要进一步检查
    # 因为 Surgery PDF 可能是文本型或扫描型
    if 'surgery' in filename:
        # 检查是否是扫描 PDF
        if DEPS['pypdf']:
            try:
                PyPDFLoader = DEPS['pypdf']
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                if docs:
                    sample_text = docs[0].page_content  # 检查第一页
                    # 几乎没有文本 -> 扫描 PDF
                    # 阈值 100 字符：文本 PDF 通常有更多文本
                    if len(sample_text.strip()) < 100:
                        return 'scanned'
                    return 'surgery'
            except:
                pass
        return 'surgery'  # 如果无法加载，假设是文本 PDF
    
    # ========================================================================
    # 步骤 2：如果文件名无法判断，分析内容
    # ========================================================================
    if not DEPS['pypdf']:
        return 'unknown'  # 没有 PyPDF，无法分析内容
    
    PyPDFLoader = DEPS['pypdf']
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 如果无法加载任何页面，可能是扫描 PDF
        if not docs:
            return 'scanned'
        
        # ====================================================================
        # 步骤 3：分析前几页的内容特征
        # ====================================================================
        # 为什么只检查前 3 页？
        #   - 提高速度（不需要加载整个文件）
        #   - 前几页通常包含格式信息
        #   - 如果前 3 页都没有文本，很可能是扫描 PDF
        sample_text = ' '.join([doc.page_content for doc in docs[:3]])
        
        # 检查 1：文本长度
        # 如果几乎没有文本（< 100 字符），很可能是扫描 PDF
        if len(sample_text.strip()) < 100:
            return 'scanned'
        
        # 检查 2：Amboss 特征标记
        # Amboss PDF 通常包含 "AMBOSS" 标记
        if 'AMBOSS' in sample_text.upper():
            return 'amboss'
        
        # 检查 3：Surgery/NBME 特征标记
        # Surgery PDF 通常包含：
        #   - "Correct Answer:" 标记
        #   - "NBME" 标记（National Board of Medical Examiners）
        if 'Correct Answer:' in sample_text or 'NBME' in sample_text.upper():
            return 'surgery'
        
        # 默认：假设是 Surgery 格式（最常见的格式）
        return 'surgery'
        
    except Exception:
        # 如果加载失败，假设是扫描 PDF（需要 OCR）
        return 'scanned'


# =============================================================================
# 主解析函数
# =============================================================================
# 
# 这些函数是解析器的入口点，负责：
#   1. 验证输入
#   2. 自动检测或使用指定的 PDF 类型
#   3. 调用相应的解析器
#   4. 处理目录批量解析
# 
# =============================================================================

def validate_question_quality(question: Dict) -> Tuple[bool, List[str]]:
    """
    验证题目解析质量，检测常见问题。
    
    功能：
        检测题目中可能存在的解析错误，如：
        1. 选项包含解释文本（选项过长或包含句子）
        2. 选项数量异常（少于 2 个或多于 10 个）
        3. 正确答案不在选项中
        4. 题目文本过短或为空
        5. 选项文本包含明显噪音（下划线、特殊字符等）
        6. 解释文本过短（可能不完整）
        7. Tip 未包含在题干中（Amboss 特有）
    
    Args:
        question: 题目字典，包含 question, options, correct_answer, explanation, tip 等字段
    
    Returns:
        tuple[bool, List[str]]: 
            - bool: True 表示质量良好，False 表示有问题
            - List[str]: 问题描述列表（如果有问题）
    
    使用场景：
        - 验证题目解析质量，检测常见问题
        - 质量检查：批量检查已解析题目的质量
    """
    issues = []
    
    # 检查基本字段
    if not question.get('question') or len(question.get('question', '').strip()) < 20:
        issues.append("题目文本过短或为空")
    
    options = question.get('options', {})
    if not isinstance(options, dict):
        issues.append("选项格式错误（不是字典）")
        return False, issues
    
    # 检查选项数量
    num_options = len(options)
    if num_options < 2:
        issues.append(f"选项数量过少（{num_options} 个）")
    elif num_options > 10:
        issues.append(f"选项数量过多（{num_options} 个）")
    
    # 检查正确答案
    correct_answer = question.get('correct_answer', '').strip().upper()
    if not correct_answer:
        issues.append("缺少正确答案")
    elif correct_answer not in options:
        issues.append(f"正确答案 '{correct_answer}' 不在选项中")
    
    # 检查每个选项的质量
    for opt_letter, opt_text in options.items():
        if not opt_text or len(opt_text.strip()) < 1:
            issues.append(f"选项 {opt_letter} 为空")
            continue
        
        opt_text_clean = opt_text.strip()
        
        # 检查选项是否过长（可能包含解释文本）
        if len(opt_text_clean) > 200:
            issues.append(f"选项 {opt_letter} 过长（{len(opt_text_clean)} 字符），可能包含解释文本")
        
        # 检查选项是否包含多个句子（可能包含解释文本）
        sentence_count = opt_text_clean.count('.') + opt_text_clean.count('!') + opt_text_clean.count('?')
        if sentence_count > 2:
            issues.append(f"选项 {opt_letter} 包含多个句子，可能包含解释文本")
        
        # 检查选项是否包含明显噪音
        if '___' in opt_text_clean or opt_text_clean.count('_') > 5:
            issues.append(f"选项 {opt_letter} 包含下划线噪音")
        
        # 检查选项是否包含常见解释文本关键词
        explanation_keywords = ['because', 'since', 'due to', 'as a result', 'therefore', 'thus', 
                                'moreover', 'furthermore', 'additionally', 'incorrect', 'correct']
        opt_lower = opt_text_clean.lower()
        if any(keyword in opt_lower for keyword in explanation_keywords) and len(opt_text_clean) > 50:
            issues.append(f"选项 {opt_letter} 可能包含解释文本（包含解释关键词）")
    
    # 检查解释文本
    explanation = question.get('explanation', '').strip()
    if not explanation:
        issues.append("解释文本为空")
    elif len(explanation) < 20:
        issues.append(f"解释文本过短（{len(explanation)} 字符），可能不完整")
    
    # 检查 Tip（对于 Amboss，tip 应该包含在题干中）
    tip = question.get('tip', '').strip()
    question_text = question.get('question', '').strip()
    if tip and tip not in question_text:
        # Tip 存在但不在题干中（Amboss 特有要求）
        issues.append("Tip 未包含在题干中（Amboss 要求 tip 应在题干中）")
    
    # 如果有任何问题，返回 False
    return len(issues) == 0, issues


def parse_pdf(pdf_path: str, pdf_type: str = 'auto', poppler_path: Optional[str] = None) -> List[Dict]:
    """
    解析单个 PDF 文件的主函数。
    
    这是解析器的核心入口点，负责：
        1. 验证文件是否存在
        2. 自动检测 PDF 类型（如果未指定）
        3. 调用相应的解析器函数
        4. 返回题目列表
    
    Args:
        pdf_path: PDF 文件路径（字符串）
        pdf_type: PDF 类型，可选值：
            - 'auto': 自动检测（默认）
            - 'surgery': 强制使用 Surgery 解析器
            - 'amboss': 强制使用 Amboss 解析器
            - 'scanned': 强制使用 OCR 解析器
        poppler_path: Poppler bin 目录路径（Windows OCR，可选）
                     注意：当前实现使用 PyMuPDF，不需要 Poppler
                     此参数保留是为了兼容性
    
    Returns:
        List[Dict]: 题目列表，格式见文件头部文档
    
    使用示例：
        # 自动检测类型
        questions = parse_pdf("Surgery 3.pdf")
        
        # 强制使用 OCR
        questions = parse_pdf("scanned.pdf", pdf_type='scanned')
    """
    # ========================================================================
    # 步骤 1：验证文件是否存在
    # ========================================================================
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return []
    
    # ========================================================================
    # 步骤 2：自动检测类型（如果需要）
    # ========================================================================
    if pdf_type == 'auto':
        pdf_type = detect_pdf_type(pdf_path)
        print(f"  Detected type: {pdf_type}")
    
    # ========================================================================
    # 步骤 3：调用相应的解析器
    # ========================================================================
    if pdf_type == 'surgery':
        return parse_surgery_pdf(pdf_path)
    elif pdf_type == 'amboss':
        return parse_amboss_pdf(pdf_path)
    elif pdf_type == 'scanned':
        return parse_scanned_pdf(pdf_path, poppler_path)
    else:
        print(f"Error: Unknown PDF type: {pdf_type}")
        return []


def parse_directory(directory: str, output_file: str = 'qbank_questions.json',
                   poppler_path: Optional[str] = None) -> List[Dict]:
    """
    批量解析目录下的所有 PDF 文件。
    
    功能：
        1. 扫描目录中的所有 PDF 文件
        2. 过滤掉只有题目没有答案的文件（Amboss 例外）
        3. 逐个解析每个 PDF
        4. 合并所有题目
        5. 保存到 JSON 文件
    
    Args:
        directory: PDF 文件目录路径
        output_file: 输出 JSON 文件路径（默认：'qbank_questions.json'）
        poppler_path: Poppler 路径（可选，用于 OCR）
    
    Returns:
        List[Dict]: 所有题目的合并列表
    
    文件过滤规则：
        - 如果文件名包含 "Questions" 但不包含 "Answers" → 跳过
        - 例外：Amboss 文件不遵循此规则，不跳过
        - 原因：某些 PDF 只有题目没有答案，无法解析
    
    处理流程：
        1. 查找所有 .pdf 文件
        2. 按文件名排序（保证处理顺序一致）
        3. 对每个文件：
           a. 检查是否需要跳过
           b. 调用 parse_pdf() 解析
           c. 合并到总列表
        4. 保存到 JSON 文件
    
    输出格式：
        JSON 数组，每个元素是一个题目对象
        使用 UTF-8 编码，确保中文正确显示
    """
    all_questions = []
    
    # ========================================================================
    # 步骤 1：查找所有 PDF 文件
    # ========================================================================
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {directory}")
    print()
    
    # ========================================================================
    # 步骤 2：逐个处理每个 PDF 文件
    # ========================================================================
    for pdf_file in sorted(pdf_files):  # 排序保证处理顺序一致
        # ====================================================================
        # 过滤：跳过只有题目没有答案的文件
        # ====================================================================
        # 某些 PDF 文件只包含题目，不包含答案和解释
        # 这些文件无法完整解析，所以跳过
        # 
        # 例外：Amboss 文件不遵循此规则
        # 因为 Amboss 格式中，答案通过百分比确定，不需要单独的答案页
        if 'Questions' in pdf_file.name and 'Answers' not in pdf_file.name:
            if 'amboss' not in pdf_file.name.lower():
                print(f"Skipping (no answers): {pdf_file.name}")
                continue
        
        # ====================================================================
        # 步骤 3：解析当前文件
        # ====================================================================
        print(f"Processing: {pdf_file.name}")
        questions = parse_pdf(str(pdf_file), poppler_path=poppler_path)
        print(f"  Parsed: {len(questions)} questions")
        all_questions.extend(questions)  # 合并到总列表
    
    # ========================================================================
    # 步骤 4：保存到 JSON 文件
    # ========================================================================
    if all_questions and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False: 允许中文字符直接写入（不转义为 \uXXXX）
            # indent=2: 缩进 2 空格，便于阅读
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_questions)} questions to {output_file}")
    
    return all_questions


# =============================================================================
# 命令行接口
# =============================================================================
# 
# 功能：提供命令行界面，方便用户使用解析器
# 
# 支持的参数：
#   - pdf: 单个 PDF 文件路径（位置参数）
#   - --dir, -d: PDF 文件目录
#   - --output, -o: 输出 JSON 文件路径
#   - --type, -t: PDF 类型（auto/surgery/amboss/scanned）
#   - --poppler: Poppler 路径（Windows OCR）
# 
# 使用场景：
#   1. 解析单个 PDF 文件
#   2. 批量解析目录下的所有 PDF
#   3. 默认解析 "Qbanks and Practice Exams" 目录
# 
# =============================================================================

def main():
    """
    命令行入口函数。
    
    解析命令行参数并调用相应的解析函数。
    
    参数说明：
        pdf: 单个 PDF 文件路径（可选位置参数）
        --dir, -d: PDF 文件目录（如果指定，解析目录下所有 PDF）
        --output, -o: 输出 JSON 文件路径（默认：qbank_questions.json）
        --type, -t: PDF 类型（auto/surgery/amboss/scanned，默认：auto）
        --poppler: Poppler bin 目录路径（Windows OCR，可选）
    
    处理逻辑：
        1. 如果指定了 --dir，解析目录下所有 PDF
        2. 如果指定了 pdf 参数，解析单个文件
        3. 如果都没有指定，尝试解析默认目录 "Qbanks and Practice Exams"
        4. 如果默认目录不存在，显示帮助信息
    """
    parser = argparse.ArgumentParser(
        description='Qbank PDF Parser - 解析医学考试题库 PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 解析单个 PDF（自动检测类型，传统方法）
  python parse_qbank.py "Surgery 3 - Answers.pdf"
  
  # 解析目录下所有 PDF
  python parse_qbank.py --dir "Qbanks and Practice Exams"
  
  # 指定输出文件
  python parse_qbank.py --dir "Qbanks and Practice Exams" -o my_questions.json
  
  # 强制使用 OCR（扫描 PDF）
  python parse_qbank.py "scanned.pdf" --type scanned
  
  # 强制使用特定解析器
  python parse_qbank.py "file.pdf" --type amboss
        """
    )
    
    # ========================================================================
    # 定义命令行参数
    # ========================================================================
    
    # 位置参数：PDF 文件路径（可选）
    parser.add_argument('pdf', nargs='?', help='PDF 文件路径（单个文件）')
    
    # 可选参数：目录
    parser.add_argument('--dir', '-d', help='PDF 文件目录（批量解析）')
    
    # 可选参数：输出文件
    parser.add_argument('--output', '-o', default='qbank_questions.json', 
                       help='输出 JSON 文件路径（默认：qbank_questions.json）')
    
    # 可选参数：PDF 类型
    parser.add_argument('--type', '-t', 
                       choices=['auto', 'surgery', 'amboss', 'scanned'],
                       default='auto', 
                       help='PDF 类型：auto（自动检测）、surgery、amboss、scanned（默认：auto）')
    
    # 可选参数：Poppler 路径（Windows OCR）
    parser.add_argument('--poppler', 
                       help='Poppler bin 目录路径（Windows OCR，可选）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # ========================================================================
    # Windows 默认 Poppler 路径（如果未指定）
    # ========================================================================
    # 注意：当前实现使用 PyMuPDF，不需要 Poppler
    # 此代码保留是为了兼容性
    poppler_path = args.poppler
    if not poppler_path and os.name == 'nt':  # os.name == 'nt' 表示 Windows
        default_poppler = r"C:\poppler\poppler-23.11.0\Library\bin"
        if os.path.exists(default_poppler):
            poppler_path = default_poppler
    
    # ========================================================================
    # 显示标题
    # ========================================================================
    print("=" * 60)
    print("Qbank PDF Parser")
    print("=" * 60)
    
    # ========================================================================
    # 根据参数选择处理方式
    # ========================================================================
    if args.dir:
        # 情况 1：指定了目录，批量解析
        parse_directory(args.dir, args.output, poppler_path)
    elif args.pdf:
        # 情况 2：指定了单个文件，解析单个文件
        questions = parse_pdf(args.pdf, args.type, poppler_path)
        if questions:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            print(f"\nSaved {len(questions)} questions to {args.output}")
    else:
        # 情况 3：没有指定参数，尝试默认目录
        qbank_dir = "Qbanks and Practice Exams"
        if os.path.exists(qbank_dir):
            parse_directory(qbank_dir, args.output, poppler_path)
        else:
            # 默认目录不存在，显示帮助信息
            parser.print_help()


if __name__ == "__main__":
    main()

