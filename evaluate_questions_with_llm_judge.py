"""
使用 LLM 作为评判者评估生成的问题质量

基于文章《使用LLM作为评判者进行自动化和多功能评估》的最佳实践：
1. 使用 1-4 的小整数刻度（而不是大的浮点数）
2. 在最终答案前添加评估字段（思考时间）
3. 提供参考标准以供指导
4. 使用结构化输出（JSON格式）

评估标准：
1. 问题清晰度：问题是否清晰、易懂
2. 选项合理性：选项是否合理、有干扰性
3. 解释充分性：解释是否充分、准确
4. 医学准确性：医学内容是否准确
5. 总体质量：综合评分

使用方法：
    # 评估 questions.json 中的问题（推荐）
    python evaluate_questions_with_llm_judge.py questions.json
    
    # 评估并计算与人类评判者的相关性
    python evaluate_questions_with_llm_judge.py questions.json --human-evaluations human_evaluated.json
    
    # 评估其他问题文件
    python evaluate_questions_with_llm_judge.py qbank_amboss_openai.json
    
    # 评估并保存结果
    python evaluate_questions_with_llm_judge.py questions.json --output questions_evaluated.json
    
    # 使用轻量模型（降低成本）
    python evaluate_questions_with_llm_judge.py questions.json --model gpt-4o-mini
    
    # 评估前 10 个问题（测试）
    python evaluate_questions_with_llm_judge.py questions.json --max-questions 10
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import pandas as pd
import numpy as np

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


def extract_judge_score(answer: str, split_str: str = "总评分：") -> Optional[float]:
    """
    从 LLM 评判者的回答中提取评分。
    
    Args:
        answer: LLM 的回答文本
        split_str: 用于分割的关键字符串
    
    Returns:
        float: 提取的评分，如果提取失败返回 None
    """
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        
        # 查找数字（支持整数和小数）
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        if digit_groups:
            return float(digit_groups[0])
        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None


def evaluate_question_with_llm_judge(
    question: Dict,
    client,
    model: str = "gpt-4o-2024-11-20"
) -> Dict:
    """
    使用 LLM 评判者评估单个问题的质量。
    
    Args:
        question: 问题字典，包含 question, options, correct_answer, explanations 等字段
        client: OpenAI 客户端
        model: 使用的模型
    
    Returns:
        Dict: 评估结果，包含各项评分和总体评分
    """
    
    # 构建改进的评判者提示（基于文章最佳实践）
    JUDGE_PROMPT = """您是一位医学教育专家，负责评估医学多选题的质量。

您将获得一个医学多选题，包括问题、选项、正确答案和解释。

请按照以下标准评估这个问题的质量，每个标准给出 1-4 分的评分：

1. **问题清晰度** (1-4分)：
   - 1分：问题模糊、难以理解，或包含歧义
   - 2分：问题基本清晰，但存在一些不明确的地方
   - 3分：问题清晰，易于理解
   - 4分：问题非常清晰、精确，表述完美

2. **选项合理性** (1-4分)：
   - 1分：选项明显不合理，或干扰项过于明显
   - 2分：部分选项合理，但干扰项质量一般
   - 3分：选项基本合理，干扰项有一定干扰性
   - 4分：所有选项都合理，干扰项具有很强的干扰性

3. **解释充分性** (1-4分)：
   - 1分：解释不充分或缺失
   - 2分：解释基本充分，但不够详细
   - 3分：解释充分，提供了合理的说明
   - 4分：解释非常充分、详细，有助于理解

4. **医学准确性** (1-4分)：
   - 1分：存在明显的医学错误
   - 2分：基本准确，但可能有小的不准确之处
   - 3分：医学内容准确
   - 4分：医学内容非常准确，符合最新医学知识

5. **总体质量** (1-4分)：
   - 1分：问题质量很差，不适合用于教学
   - 2分：问题质量一般，需要改进
   - 3分：问题质量良好，可以用于教学
   - 4分：问题质量优秀，非常适合用于教学

请按以下格式提供您的反馈：

评估：：
（请用文字详细描述您的评分理由，包括每个标准的评估）

问题清晰度：<1-4之间的整数>
选项合理性：<1-4之间的整数>
解释充分性：<1-4之间的整数>
医学准确性：<1-4之间的整数>
总体质量：<1-4之间的整数>

总评分：<1-4之间的整数>

以下是需要评估的问题：

问题：{question_text}

选项：
{options_text}

正确答案：{correct_answer}

解释：
{explanations_text}

请提供您的评估。如果您给出准确、公正的评分，我将非常感激。

评估：：
问题清晰度：
选项合理性：
解释充分性：
医学准确性：
总体质量：
总评分："""

    # 格式化问题内容
    question_text = question.get('question', '')
    
    # 格式化选项
    options = question.get('options', {})
    options_text = '\n'.join([f"{k}: {v}" for k, v in options.items()])
    
    # 正确答案
    correct_answer = question.get('correct_answer', '')
    
    # 格式化解释
    explanations = question.get('explanations', {})
    explanations_text = '\n'.join([f"{k}: {v}" for k, v in explanations.items() if v])
    
    # 构建完整提示
    full_prompt = JUDGE_PROMPT.format(
        question_text=question_text,
        options_text=options_text,
        correct_answer=correct_answer,
        explanations_text=explanations_text
    )
    
    try:
        # 调用 OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "您是一位专业的医学教育评估专家。"},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,  # 降低温度以获得更一致的评估
            max_tokens=1000
        )
        
        judge_response = response.choices[0].message.content
        
        # 解析评估结果
        evaluation_result = {
            "evaluation_text": judge_response,
            "question_clarity": extract_judge_score(judge_response, "问题清晰度：") or extract_judge_score(judge_response, "问题清晰度"),
            "option_quality": extract_judge_score(judge_response, "选项合理性：") or extract_judge_score(judge_response, "选项合理性"),
            "explanation_quality": extract_judge_score(judge_response, "解释充分性：") or extract_judge_score(judge_response, "解释充分性"),
            "medical_accuracy": extract_judge_score(judge_response, "医学准确性：") or extract_judge_score(judge_response, "医学准确性"),
            "overall_quality": extract_judge_score(judge_response, "总体质量：") or extract_judge_score(judge_response, "总体质量"),
            "total_score": extract_judge_score(judge_response, "总评分：") or extract_judge_score(judge_response, "总评分")
        }
        
        return evaluation_result
        
    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            "evaluation_text": f"Error: {str(e)}",
            "question_clarity": None,
            "option_quality": None,
            "explanation_quality": None,
            "medical_accuracy": None,
            "overall_quality": None,
            "total_score": None
        }


def evaluate_questions_batch(
    questions: List[Dict],
    client,
    model: str = "gpt-4o-2024-11-20",
    start_index: int = 0,
    max_questions: Optional[int] = None
) -> List[Dict]:
    """
    批量评估问题。
    
    Args:
        questions: 问题列表
        client: OpenAI 客户端
        model: 使用的模型
        start_index: 开始评估的索引
        max_questions: 最多评估的问题数量（None 表示评估所有）
    
    Returns:
        List[Dict]: 评估结果列表，每个结果包含原始问题和评估信息
    """
    results = []
    
    end_index = len(questions)
    if max_questions:
        end_index = min(start_index + max_questions, len(questions))
    
    questions_to_evaluate = questions[start_index:end_index]
    
    print(f"Evaluating {len(questions_to_evaluate)} questions (indices {start_index} to {end_index-1})...")
    
    for i, question in enumerate(tqdm(questions_to_evaluate, desc="Evaluating")):
        # 添加延迟以避免 API 速率限制
        if i > 0:
            time.sleep(0.5)
        
        evaluation = evaluate_question_with_llm_judge(question, client, model)
        
        # 合并原始问题和评估结果
        result = {
            **question,
            "llm_judge_evaluation": evaluation
        }
        
        results.append(result)
    
    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    计算评估统计信息。
    
    Args:
        results: 评估结果列表
    
    Returns:
        Dict: 统计信息
    """
    stats = {
        "total_questions": len(results),
        "question_clarity": [],
        "option_quality": [],
        "explanation_quality": [],
        "medical_accuracy": [],
        "overall_quality": [],
        "total_score": []
    }
    
    for result in results:
        eval_data = result.get("llm_judge_evaluation", {})
        
        if eval_data.get("question_clarity") is not None:
            stats["question_clarity"].append(eval_data["question_clarity"])
        if eval_data.get("option_quality") is not None:
            stats["option_quality"].append(eval_data["option_quality"])
        if eval_data.get("explanation_quality") is not None:
            stats["explanation_quality"].append(eval_data["explanation_quality"])
        if eval_data.get("medical_accuracy") is not None:
            stats["medical_accuracy"].append(eval_data["medical_accuracy"])
        if eval_data.get("overall_quality") is not None:
            stats["overall_quality"].append(eval_data["overall_quality"])
        if eval_data.get("total_score") is not None:
            stats["total_score"].append(eval_data["total_score"])
    
    # 计算平均值
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    statistics = {
        "total_questions": stats["total_questions"],
        "average_question_clarity": avg(stats["question_clarity"]),
        "average_option_quality": avg(stats["option_quality"]),
        "average_explanation_quality": avg(stats["explanation_quality"]),
        "average_medical_accuracy": avg(stats["medical_accuracy"]),
        "average_overall_quality": avg(stats["overall_quality"]),
        "average_total_score": avg(stats["total_score"]),
        "questions_evaluated": {
            "question_clarity": len(stats["question_clarity"]),
            "option_quality": len(stats["option_quality"]),
            "explanation_quality": len(stats["explanation_quality"]),
            "medical_accuracy": len(stats["medical_accuracy"]),
            "overall_quality": len(stats["overall_quality"]),
            "total_score": len(stats["total_score"])
        }
    }
    
    return statistics


def calculate_correlation_with_human(
    llm_evaluations: List[Dict],
    human_evaluations_file: str
) -> Dict[str, float]:
    """
    计算 LLM 评判者与人类评判者之间的皮尔逊相关系数。
    
    这是评估 LLM 评判者可靠性的关键指标。
    
    Args:
        llm_evaluations: LLM 评估结果列表
        human_evaluations_file: 人类评估文件路径
    
    Returns:
        Dict: 各维度的皮尔逊相关系数
    """
    print(f"\n{'='*60}")
    print("计算 LLM 评判者与人类评判者之间的相关性")
    print(f"{'='*60}\n")
    
    # 加载人类评估
    try:
        with open(human_evaluations_file, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载人类评估文件: {e}")
        return {}
    
    # 提取人类评估（支持多种格式）
    human_evaluations = {}
    if 'evaluations' in human_data:
        human_evaluations = human_data['evaluations']
    elif 'questions' in human_data:
        for i, q in enumerate(human_data['questions']):
            if 'human_evaluation' in q and q['human_evaluation']:
                human_evaluations[i] = q['human_evaluation']
    
    if not human_evaluations:
        print("错误: 人类评估文件中没有找到评估数据")
        return {}
    
    print(f"加载了 {len(human_evaluations)} 个人类评估")
    
    # 提取 LLM 评估
    llm_scores_by_dim = {
        'question_clarity': {},
        'option_quality': {},
        'explanation_quality': {},
        'medical_accuracy': {},
        'overall_quality': {},
        'total_score': {}
    }
    
    for i, result in enumerate(llm_evaluations):
        eval_data = result.get('llm_judge_evaluation', {})
        for dim in llm_scores_by_dim.keys():
            score = eval_data.get(dim)
            if score is not None:
                llm_scores_by_dim[dim][i] = score
    
    # 计算每个维度的相关性
    dimensions = ['question_clarity', 'option_quality', 'explanation_quality', 
                  'medical_accuracy', 'overall_quality', 'total_score']
    
    correlations = {}
    
    for dim in dimensions:
        # 找到共同评估的问题索引
        llm_scores = llm_scores_by_dim[dim]
        human_scores = {}
        
        for idx, eval_data in human_evaluations.items():
            score = eval_data.get(dim)
            if score is not None:
                human_scores[int(idx)] = score
        
        # 找到共同的问题索引
        common_indices = set(llm_scores.keys()) & set(human_scores.keys())
        
        if len(common_indices) < 2:
            correlations[dim] = None
            print(f"  {dim:25s}: N/A (共同评估的问题数不足: {len(common_indices)})")
            continue
        
        # 构建评分列表（按索引排序）
        llm_values = [llm_scores[idx] for idx in sorted(common_indices)]
        human_values = [human_scores[idx] for idx in sorted(common_indices)]
        
        # 计算皮尔逊相关系数
        df = pd.DataFrame({
            'llm': llm_values,
            'human': human_values
        })
        
        corr = df['llm'].corr(df['human'], method='pearson')
        correlations[dim] = corr if not pd.isna(corr) else None
        
        if corr is not None:
            print(f"  {dim:25s}: {corr:.3f} (n={len(common_indices)})")
        else:
            print(f"  {dim:25s}: N/A (无法计算)")
    
    print(f"\n{'='*60}\n")
    
    return correlations


def main():
    parser = argparse.ArgumentParser(
        description='使用 LLM 作为评判者评估医学问题质量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 评估所有问题
  python evaluate_questions_with_llm_judge.py qbank_amboss_openai.json
  
  # 评估前 10 个问题（测试）
  python evaluate_questions_with_llm_judge.py qbank_amboss_openai.json --max-questions 10
  
  # 使用轻量模型
  python evaluate_questions_with_llm_judge.py qbank_amboss_openai.json --model gpt-4o-mini
  
  # 保存评估结果
  python evaluate_questions_with_llm_judge.py qbank_amboss_openai.json --output evaluation_results.json
        """
    )
    
    parser.add_argument('input_file', type=str,
                       help='输入的问题 JSON 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出评估结果 JSON 文件路径（默认：在输入文件名后添加 _evaluated）')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API 密钥（如果不提供，尝试从环境变量或 api-key.js 读取）')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20',
                       help='OpenAI 模型（默认：gpt-4o-2024-11-20。可选：gpt-5, gpt-4o-2024-11-20, gpt-4o, gpt-4o-mini）')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='最多评估的问题数量（默认：评估所有问题）')
    parser.add_argument('--start-index', type=int, default=0,
                       help='开始评估的问题索引（默认：0）')
    parser.add_argument('--human-evaluations', type=str, default=None,
                       help='人类评估文件路径（用于计算相关性）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # 获取 API 密钥
    print("Getting API key...")
    api_key = get_openai_api_key(args.api_key)
    if not api_key:
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide --api-key parameter.")
        return
    print("API key found.")
    
    # 初始化 OpenAI 客户端
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: OpenAI library not installed. Run: pip install openai")
        return
    
    client = OpenAI(api_key=api_key)
    
    # 加载问题
    print(f"Loading questions from {args.input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions.")
    
    # 评估问题
    results = evaluate_questions_batch(
        questions,
        client,
        model=args.model,
        start_index=args.start_index,
        max_questions=args.max_questions
    )
    
    # 计算统计信息（仅用于了解整体质量分布，不是对单个问题的多次评估取平均）
    statistics = calculate_statistics(results)
    
    print("\n" + "="*60)
    print("评估统计信息")
    print("="*60)
    print(f"总问题数: {statistics['total_questions']}")
    print(f"已评估问题数: {statistics['questions_evaluated']['total_score']}")
    print(f"\n整体质量分布（所有问题的平均分，用于了解整体质量）:")
    print(f"  问题清晰度: {statistics['average_question_clarity']:.2f}/4.0")
    print(f"  选项合理性: {statistics['average_option_quality']:.2f}/4.0")
    print(f"  解释充分性: {statistics['average_explanation_quality']:.2f}/4.0")
    print(f"  医学准确性: {statistics['average_medical_accuracy']:.2f}/4.0")
    print(f"  总体质量: {statistics['average_overall_quality']:.2f}/4.0")
    print(f"  总评分: {statistics['average_total_score']:.2f}/4.0")
    print("\n注意：每个问题只评估一次，上述平均分用于了解整体质量分布")
    print("="*60)
    
    # 如果提供了人类评估文件，计算相关性
    correlations = {}
    if args.human_evaluations:
        correlations = calculate_correlation_with_human(results, args.human_evaluations)
        # 将相关性添加到统计信息中
        statistics['llm_human_correlations'] = correlations
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_evaluated{input_path.suffix}"
    
    output_data = {
        "statistics": statistics,
        "questions": results
    }
    
    print(f"\nSaving evaluation results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Evaluation complete! Results saved to {output_path}")


if __name__ == '__main__':
    main()

