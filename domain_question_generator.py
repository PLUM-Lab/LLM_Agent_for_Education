"""
================================================================================
Domain → Topic → Subtopic Question Generator (Medical Students)
================================================================================

功能：
    - 从一个 JSON 配置文件读取：领域 → 主题 → 子主题
    - 针对每个「子主题」调用 OpenAI，让 ChatGPT 生成若干面向医学生的题目
    - 输出统一的 JSONL 文件，后续可导入测验系统

JSON 配置结构示例（domains_config.json）：

{
  "domains": [
    {
      "name": "Cardiology",
      "level": "MS2",
      "topics": [
        {
          "name": "Heart Failure",
          "subtopics": [
            "Pathophysiology",
            "Clinical Presentation",
            "Pharmacologic Management"
          ]
        },
        {
          "name": "Acute Coronary Syndromes",
          "subtopics": [
            "STEMI Diagnosis",
            "NSTEMI Risk Stratification"
          ]
        }
      ]
    }
  ]
}

运行方式：
    1. 在项目根目录创建 domains_config.json（结构如上）
    2. 配置 OPENAI_API_KEY 环境变量或使用现有 api-key.js
    3. 运行：
           python domain_question_generator.py
    4. 生成文件：
           generated_domain_questions.jsonl

注意：
    - 默认每个子主题生成 3 个题目（可在 MAIN_CONFIG 中修改）
    - 题目面向医学生（clinical vignette + single best answer）
================================================================================
"""

from __future__ import annotations  # 启用类型注解的前向引用

import json  # JSON文件处理
import os  # 操作系统接口（用于环境变量）
from dataclasses import dataclass, asdict  # 数据类和序列化
from pathlib import Path  # 路径处理
from typing import List, Dict, Any, Optional  # 类型提示
import re  # 正则表达式（用于从api-key.js提取密钥）

try:
    from openai import OpenAI  # OpenAI API客户端
except ImportError:
    # 复用项目里安装 openai 的逻辑
    # 如果未安装，自动安装openai包
    import subprocess

    subprocess.check_call(["pip", "install", "openai"])
    from openai import OpenAI


# =============================================================================
# 配置
# Configuration
# =============================================================================

MAIN_CONFIG = {
    # 使用经过整理的完整领域→主题→子主题文件
    # 你已经在项目根目录有：all_domains_with_subtopics.json
    # 使用经过整理的完整领域→主题→子主题文件
    # 输入配置文件路径：包含领域、主题、子主题的层次结构
    "domains_config_path": "all_domains_with_subtopics.json",
    
    # 输出文件路径（已改为JSON格式，不再是JSONL）
    # 输出文件路径：生成的题目将保存为JSON数组格式
    "output_path": "generated_domain_questions.json",  # Changed to .json format
    
    # 每个子主题生成的题目数量
    # 每个子主题生成的题目数量：默认3题，可根据需要调整
    "questions_per_subtopic": 3,
    
    # 使用的OpenAI模型
    # 使用的OpenAI模型：gpt-4o提供更好的题目质量
    "model": "gpt-4o",
}


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class GeneratedQuestion:
    """
    生成后的题目结构。
    Generated question data structure.

    说明：
    Description:
        - 前三项 domain/topic/subtopic 是方便你后续按领域筛选、分桶。
          前三个字段（domain/topic/subtopic）用于后续按领域筛选和分类
        - question/options/correct_answer/explanations 与 UI 中 questions_evaluated.json 的结构兼容：
          问题、选项、正确答案、解释字段与UI中的questions_evaluated.json结构兼容：
              question: 问题文本
              options: { "A": "...", "B": "...", "C": "...", "D": "..." } 选项字典
              correct_answer: "A" 正确答案字母
              explanations: { "A": "...", "B": "...", "C": "...", "D": "..." } 每个选项的解释
    """

    domain: str  # 领域名称，如 "Cardiology"（领域名称）
    topic: str  # 主题名称，如 "Heart Failure"（主题名称）
    subtopic: str  # 子主题名称，如 "Pathophysiology"（子主题名称）
    difficulty: str  # e.g. "easy", "medium", "hard" 难度级别（简单/中等/困难）
    question: str  # 问题文本（完整的问题描述，可能包含临床场景）
    options: Dict[str, str]  # 选项字典，格式：{"A": "选项A文本", "B": "选项B文本", ...}
    correct_answer: str  # "A" / "B" / ... 正确答案的字母（A、B、C或D）
    explanations: Dict[str, str]  # per-option explanations 每个选项的解释字典，格式：{"A": "解释A", "B": "解释B", ...}


# =============================================================================
# 工具函数
# Utility Functions
# =============================================================================

def get_api_key() -> str:
    """
    从环境变量或 api-key.js 中获取 OpenAI API Key。
    Get OpenAI API key from environment variable or api-key.js file.
    
    优先级：
    Priority:
    1. 环境变量 OPENAI_API_KEY（最高优先级）
       环境变量 OPENAI_API_KEY (highest priority)
    2. api-key.js 文件中的配置（如果环境变量未设置）
       Configuration in api-key.js file (if environment variable not set)
    
    Returns:
        返回：API密钥字符串，如果未找到则返回空字符串
        str: API key string, or empty string if not found
    """
    # 首先尝试从环境变量获取
    # Try environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 如果环境变量中没有，尝试从api-key.js文件读取
    # If not in environment variable, try reading from api-key.js file
    config_path = Path(__file__).parent / "api-key.js"
    if config_path.exists():
        try:
            content = config_path.read_text(encoding="utf-8")
            # 使用正则表达式提取API密钥（支持多种格式：= 或 :，单引号或双引号）
            # Extract API key using regex (supports various formats: = or :, single or double quotes)
            match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
            if match:
                return match.group(1)
        except Exception as e:
            print(f"[WARN] 读取 api-key.js 失败: {e}")
    return ""


def load_domains_config(path: str) -> Dict[str, Any]:
    """
    加载领域配置 JSON。
    Load domain configuration JSON file.
    
    Args:
        参数：
        path (str): 配置文件路径
            Configuration file path
    
    Returns:
        返回：
        Dict[str, Any]: 解析后的JSON配置字典
            Parsed JSON configuration dictionary
    
    Raises:
        异常：
        FileNotFoundError: 如果配置文件不存在
            If configuration file does not exist
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Domains config file not found: {cfg_path.resolve()}\n"
            f"请创建一个 JSON 文件，结构类似于文档顶部示例。"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)




# =============================================================================
# Prompt 模板（面向医学生的题目生成）
# Prompt Templates (for Medical Student Question Generation)
# =============================================================================

QUESTION_GEN_SYSTEM_PROMPT = """You are a medical education expert and clinical question writer.

Your task is to generate high-quality exam questions for medical students.

Constraints:
- Audience: medical students (pre-clinical and clinical, depending on domain level)
  受众：医学生（根据领域级别，可能是临床前或临床阶段）
- Format: USMLE-style single best answer (MCQ) with a brief clinical vignette when appropriate
  格式：USMLE风格的单选最佳答案（MCQ），适当情况下包含简短临床场景
- Focus: conceptual understanding + clinical reasoning, not pure memorization
  重点：概念理解 + 临床推理，而非纯记忆
- Domain: clinical medicine (no basic math / generic logic puzzles)
  领域：临床医学（不包含基础数学/通用逻辑题）
- DO NOT reveal the correct answer in the question stem.
  不要在问题题干中透露正确答案。
"""


def build_question_user_prompt(
    domain: str, topic: str, subtopic: str, questions_per_subtopic: int
) -> str:
    """
    构建用户 prompt，告诉模型生成该子主题的若干题目。
    Build user prompt to instruct the model to generate questions for a specific subtopic.
    
    Args:
        参数：
        domain (str): 领域名称，如 "Cardiology"
            领域名称
        topic (str): 主题名称，如 "Heart Failure"
            主题名称
        subtopic (str): 子主题名称，如 "Pathophysiology"
            子主题名称
        questions_per_subtopic (int): 每个子主题要生成的题目数量
            每个子主题要生成的题目数量
    
    Returns:
        返回：
        str: 格式化后的用户提示词
             Formatted user prompt string
    """
    return f"""Generate exam questions for medical students in JSON format only (no extra text).

Domain: {domain}
Topic: {topic}
Subtopic: {subtopic}

Number of questions: {questions_per_subtopic}

Requirements:
- Each question should clearly target the subtopic: "{subtopic}"
- Use clinical vignettes when appropriate
- Difficulty: mix of easy/medium, occasional hard, but always fair
- Each question must have 4 options: A, B, C, D
- Exactly ONE best answer

Explanation format (IMPORTANT, to match existing UI schema):
- Provide a separate explanation for EACH option (A, B, C, D)
- Explanations should briefly state why that specific option is correct or incorrect
- Explanations should be at a medical student level (not attending-level depth)

Return a JSON object with the following structure:
{{
  "questions": [
    {{
      "question": "Full question text...",
      "difficulty": "easy | medium | hard",
      "options": {{
        "A": "option A text",
        "B": "option B text",
        "C": "option C text",
        "D": "option D text"
      }},
      "correct_answer": "A",
      "explanations": {{
        "A": "Why A is correct or why it is wrong",
        "B": "Why B is correct or why it is wrong",
        "C": "Why C is correct or why it is wrong",
        "D": "Why D is correct or why it is wrong"
      }}
    }}
  ]
}}
"""


# =============================================================================
# 主逻辑
# =============================================================================

def generate_questions_for_subtopic(
    client: OpenAI,
    model: str,
    domain: str,
    topic: str,
    subtopic: str,
    questions_per_subtopic: int,
) -> List[GeneratedQuestion]:
    """
    调用 OpenAI 为单个子主题生成题目列表。
    Call OpenAI API to generate a list of questions for a single subtopic.
    
    Args:
        参数：
        client (OpenAI): OpenAI API客户端实例
            OpenAI API client instance
        model (str): 使用的模型名称，如 "gpt-4o"
            使用的模型名称
        domain (str): 领域名称
            领域名称
        topic (str): 主题名称
            主题名称
        subtopic (str): 子主题名称
            子主题名称
        questions_per_subtopic (int): 每个子主题要生成的题目数量
            每个子主题要生成的题目数量
    
    Returns:
        返回：
        List[GeneratedQuestion]: 生成的题目对象列表
            List of generated question objects
    
    Raises:
        异常：
        ValueError: 如果模型输出不是有效的JSON格式
            If model output is not valid JSON format
    """
    # 构建用户提示词
    # Build user prompt
    user_prompt = build_question_user_prompt(
        domain, topic, subtopic, questions_per_subtopic
    )

    # 调用OpenAI API生成题目
    # Call OpenAI API to generate questions
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,  # 温度参数：控制输出的随机性（0.7提供适度的创造性）
        max_tokens=2000,  # 最大token数：限制响应长度
        response_format={"type": "json_object"},  # 强制返回JSON格式
    )

    # 解析API响应
    # Parse API response
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"模型输出不是合法 JSON: {content[:200]}...")

    # 提取题目数据
    # Extract questions data
    questions_data = data.get("questions", [])
    results: List[GeneratedQuestion] = []

    # 验证和清理每个题目
    # Validate and clean each question
    for q in questions_data:
        options = q.get("options", {})
        # 只保留 A-D 选项（确保只有4个选项）
        # Only keep A-D options (ensure exactly 4 options)
        options_clean = {k: v for k, v in options.items() if k in ["A", "B", "C", "D"]}
        if len(options_clean) != 4:
            # 跳过不规范的题目（选项数量不是4个）
            # Skip invalid questions (not exactly 4 options)
            continue

        correct = q.get("correct_answer", "").strip()
        if correct not in options_clean:
            # 跳过没有合法答案的题目（正确答案不在选项列表中）
            # Skip questions without valid answer (correct answer not in options)
            continue

        explanations_raw = q.get("explanations", {})
        # 确保 explanations 至少覆盖 A-D，即使是空字符串也可以，方便 UI 处理
        # Ensure explanations cover at least A-D, even if empty string, for UI compatibility
        explanations_clean: Dict[str, str] = {}
        for key in ["A", "B", "C", "D"]:
            text = ""
            if isinstance(explanations_raw, dict):
                text = str(explanations_raw.get(key, "")).strip()
            explanations_clean[key] = text

        # 创建题目对象并添加到结果列表
        # Create question object and add to results list
        results.append(
            GeneratedQuestion(
                domain=domain,
                topic=topic,
                subtopic=subtopic,
                difficulty=q.get("difficulty", "medium"),  # 默认难度为中等
                question=q.get("question", "").strip(),  # 去除首尾空白
                options=options_clean,
                correct_answer=correct,
                explanations=explanations_clean,
            )
        )

    return results


def main():
    """
    主函数：遍历所有领域、主题、子主题，生成题目并保存到JSON文件。
    Main function: Iterate through all domains, topics, and subtopics to generate questions and save to JSON file.
    
    流程：
    Process:
    1. 加载配置和API密钥
       加载配置和API密钥
    2. 读取领域配置文件
       读取领域配置文件
    3. 遍历每个领域 → 主题 → 子主题
       遍历每个领域 → 主题 → 子主题
    4. 为每个子主题调用OpenAI生成题目
       为每个子主题调用OpenAI生成题目
    5. 将所有题目保存为JSON数组格式
       将所有题目保存为JSON数组格式
    """
    cfg = MAIN_CONFIG

    # 获取API密钥
    # Get API key
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. 请设置环境变量 OPENAI_API_KEY 或在 api-key.js 中配置。"
        )

    # 初始化OpenAI客户端
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # 加载领域配置
    # Load domain configuration
    print("=== Loading domain config:", cfg["domains_config_path"], "===")
    domains_cfg = load_domains_config(cfg["domains_config_path"])
    domains = domains_cfg.get("domains", [])
    if not domains:
        print("[WARN] No domains found in config.")
        return

    # 处理输出文件路径（确保是.json格式，不是.jsonl）
    # Process output file path (ensure .json format, not .jsonl)
    output_path = Path(cfg["output_path"])
    if output_path.suffix == ".jsonl":
        # 如果配置的是.jsonl，改为.json
        # If configured as .jsonl, change to .json
        output_path = output_path.with_suffix(".json")
    
    total_generated = 0  # 总生成题目数
    all_questions = []  # Collect all questions first 收集所有题目（先收集，最后统一写入）

    # 三层循环：领域 → 主题 → 子主题
    # Three-level loop: Domain → Topic → Subtopic
    for dom in domains:
        domain_name = dom.get("name", "Unknown Domain")
        topics = dom.get("topics", [])
        print(f"\n=== Domain: {domain_name} ===")

        for topic in topics:
            topic_name = topic.get("name", "Unknown Topic")
            subtopics = topic.get("subtopics", [])
            print(f"  - Topic: {topic_name} ({len(subtopics)} subtopics)")

            for sub in subtopics:
                sub_name = sub if isinstance(sub, str) else str(sub)
                print(f"    * Subtopic: {sub_name} ... ", end="", flush=True)

                try:
                    # 为当前子主题生成题目
                    # Generate questions for current subtopic
                    gen_qs = generate_questions_for_subtopic(
                        client=client,
                        model=cfg["model"],
                        domain=domain_name,
                        topic=topic_name,
                        subtopic=sub_name,
                        questions_per_subtopic=cfg["questions_per_subtopic"],
                    )
                    # Convert to dict and add to list
                    # 将题目对象转换为字典并添加到列表
                    for q in gen_qs:
                        all_questions.append(asdict(q))  # asdict将dataclass转换为字典
                    total_generated += len(gen_qs)
                    print(f"OK ({len(gen_qs)} questions)")
                except Exception as e:
                    # 如果生成失败，打印错误但继续处理下一个子主题
                    # If generation fails, print error but continue with next subtopic
                    print(f"ERROR: {e}")

    # Write all questions as a JSON array
    # 将所有题目写入JSON数组格式的文件
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(
            all_questions, 
            fout, 
            ensure_ascii=False,  # 允许中文字符（不转义为\\u格式）
            indent=2  # 缩进2个空格，便于阅读
        )

    print(
        f"\nDone. Generated {total_generated} questions → {output_path.resolve()}"
    )


if __name__ == "__main__":
    # 当直接运行此脚本时，执行主函数
    # When running this script directly, execute main function
    main()


