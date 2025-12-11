"""
合并多个用户审阅者的评估结果，只保留意见一致的评估

基于文章《使用LLM作为评判者进行自动化和多功能评估》的方法：
- 只选择人工审阅者意见一致的样本
- 计算审阅者之间的一致性（皮尔逊相关性）
- 降低评估噪声

使用方法：
    # 合并多个评估文件
    python merge_human_evaluations.py --files evaluator1.json evaluator2.json evaluator3.json --output merged_agreed.json
    
    # 指定一致性阈值（允许的差异范围）
    python merge_human_evaluations.py --files eval1.json eval2.json --threshold 0 --output agreed.json
    
    # 显示统计信息
    python merge_human_evaluations.py --files eval1.json eval2.json --stats-only
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

def load_evaluation_file(file_path: str) -> Dict:
    """
    加载评估文件。
    
    Args:
        file_path: 评估文件路径
    
    Returns:
        Dict: 包含问题和评估数据的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式：
    # 1. {"questions": [...], "evaluations": {...}}
    # 2. {"statistics": {...}, "questions": [...]}
    if 'evaluations' in data:
        return {
            'questions': data.get('questions', []),
            'evaluations': data['evaluations']
        }
    elif 'questions' in data:
        # 从 questions 中提取评估
        evaluations = {}
        for i, q in enumerate(data['questions']):
            if 'human_evaluation' in q and q['human_evaluation']:
                evaluations[i] = q['human_evaluation']
            elif 'llm_judge_evaluation' in q and q['llm_judge_evaluation']:
                evaluations[i] = q['llm_judge_evaluation']
        return {
            'questions': data['questions'],
            'evaluations': evaluations
        }
    else:
        raise ValueError(f"无法识别文件格式: {file_path}")

def extract_scores(evaluation: Dict) -> Dict[str, Optional[float]]:
    """
    从评估中提取各项评分。
    
    Args:
        evaluation: 评估字典
    
    Returns:
        Dict: 各项评分的字典
    """
    return {
        'question_clarity': evaluation.get('question_clarity'),
        'option_quality': evaluation.get('option_quality'),
        'explanation_quality': evaluation.get('explanation_quality'),
        'medical_accuracy': evaluation.get('medical_accuracy'),
        'overall_quality': evaluation.get('overall_quality'),
        'total_score': evaluation.get('total_score')
    }

def scores_agree(score1: Optional[float], score2: Optional[float], threshold: float = 0) -> bool:
    """
    判断两个评分是否一致。
    
    Args:
        score1: 第一个评分
        score2: 第二个评分
        threshold: 允许的差异阈值（0表示必须完全一致）
    
    Returns:
        bool: 是否一致
    """
    if score1 is None or score2 is None:
        return False
    return abs(score1 - score2) <= threshold

def all_scores_agree(scores_list: List[Dict[str, Optional[float]]], threshold: float = 0) -> bool:
    """
    判断所有审阅者的评分是否一致。
    
    Args:
        scores_list: 所有审阅者的评分列表
        threshold: 允许的差异阈值
    
    Returns:
        bool: 是否所有评分都一致
    """
    if len(scores_list) < 2:
        return True
    
    # 检查每个维度
    dimensions = ['question_clarity', 'option_quality', 'explanation_quality', 
                  'medical_accuracy', 'overall_quality', 'total_score']
    
    for dim in dimensions:
        values = [s.get(dim) for s in scores_list if s.get(dim) is not None]
        if len(values) < 2:
            continue
        
        # 检查是否所有值都一致（在阈值范围内）
        first_value = values[0]
        for val in values[1:]:
            if abs(first_value - val) > threshold:
                return False
    
    return True

def calculate_correlation(evaluations_list: List[Dict]) -> Dict[str, float]:
    """
    计算审阅者之间的皮尔逊相关性。
    
    Args:
        evaluations_list: 所有审阅者的评估列表
    
    Returns:
        Dict: 各维度的相关性字典
    """
    if len(evaluations_list) < 2:
        return {}
    
    dimensions = ['question_clarity', 'option_quality', 'explanation_quality', 
                  'medical_accuracy', 'overall_quality', 'total_score']
    
    correlations = {}
    
    for dim in dimensions:
        # 为每个审阅者提取该维度的评分
        scores_by_evaluator = []
        for eval_data in evaluations_list:
            scores = []
            for q_idx, evaluation in eval_data.items():
                score = evaluation.get(dim)
                if score is not None:
                    scores.append((q_idx, score))
            scores_by_evaluator.append(dict(scores))
        
        # 找到所有审阅者都评估了的问题
        common_indices = set(scores_by_evaluator[0].keys())
        for scores_dict in scores_by_evaluator[1:]:
            common_indices &= set(scores_dict.keys())
        
        if len(common_indices) < 2:
            correlations[dim] = None
            continue
        
        # 构建 DataFrame 计算相关性
        data = {}
        for i, scores_dict in enumerate(scores_by_evaluator):
            data[f'evaluator_{i+1}'] = [scores_dict[idx] for idx in sorted(common_indices)]
        
        df = pd.DataFrame(data)
        corr = df.corr(method='pearson')
        
        # 取所有审阅者之间的平均相关性
        if len(corr) > 1:
            # 获取上三角矩阵的平均值（排除对角线）
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            avg_corr = corr.where(mask).stack().mean()
            correlations[dim] = avg_corr if not pd.isna(avg_corr) else None
        else:
            correlations[dim] = None
    
    return correlations

def merge_evaluations(
    evaluation_files: List[str],
    threshold: float = 0,
    output_file: Optional[str] = None,
    stats_only: bool = False,
    keep_all_evaluators: bool = False
) -> Dict:
    """
    合并多个评估文件，只保留意见一致的评估。
    
    Args:
        evaluation_files: 评估文件路径列表
        threshold: 允许的评分差异阈值（0表示必须完全一致）
        output_file: 输出文件路径
        stats_only: 是否只显示统计信息，不保存结果
        keep_all_evaluators: 是否保留所有审阅者的评估（默认False，只保留一个，因为评分一样）
    
    Returns:
        Dict: 合并后的结果
    """
    print(f"\n{'='*60}")
    print(f"合并 {len(evaluation_files)} 个评估文件")
    print(f"{'='*60}\n")
    
    # 加载所有评估文件
    all_evaluations = []
    all_questions = []
    
    for i, file_path in enumerate(evaluation_files):
        print(f"加载评估文件 {i+1}: {file_path}")
        try:
            data = load_evaluation_file(file_path)
            all_evaluations.append(data['evaluations'])
            if not all_questions:
                all_questions = data['questions']
            print(f"  ✓ 加载成功: {len(data['evaluations'])} 个评估")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue
    
    if len(all_evaluations) < 2:
        print("\n错误: 需要至少 2 个评估文件才能进行合并")
        return {}
    
    # 找到所有评估文件都包含的问题索引
    common_indices = set(all_evaluations[0].keys())
    for eval_data in all_evaluations[1:]:
        common_indices &= set(eval_data.keys())
    
    print(f"\n共同评估的问题数: {len(common_indices)}")
    
    # 计算相关性
    print("\n计算审阅者之间的相关性...")
    correlations = calculate_correlation(all_evaluations)
    
    print("\n审阅者之间的相关性（皮尔逊相关系数）:")
    print("-" * 60)
    for dim, corr in correlations.items():
        if corr is not None:
            print(f"  {dim:25s}: {corr:.3f}")
        else:
            print(f"  {dim:25s}: N/A (数据不足)")
    
    # 找出意见一致的问题
    print(f"\n查找意见一致的问题（阈值: {threshold}）...")
    agreed_indices = []
    
    for q_idx in sorted(common_indices):
        # 提取所有审阅者对该问题的评分
        scores_list = []
        for eval_data in all_evaluations:
            if q_idx in eval_data:
                scores = extract_scores(eval_data[q_idx])
                scores_list.append(scores)
        
        # 检查是否一致
        if all_scores_agree(scores_list, threshold):
            agreed_indices.append(q_idx)
    
    print(f"✓ 找到 {len(agreed_indices)} 个意见一致的问题（共 {len(common_indices)} 个）")
    print(f"  一致率: {len(agreed_indices)/len(common_indices)*100:.1f}%")
    
    # 构建合并后的结果
    merged_evaluations = {}
    merged_questions = []
    
    for q_idx in agreed_indices:
        # 如果所有审阅者意见一致，评分是一样的
        # 可以选择只保留一个评估结果，或者保留所有审阅者的评估（评估文本可能不同）
        if keep_all_evaluators:
            # 选项：保留所有审阅者的评估（评估文本可能不同）
            all_evaluators_scores = []
            for i, eval_data in enumerate(all_evaluations):
                if q_idx in eval_data:
                    all_evaluators_scores.append({
                        'evaluator_id': i + 1,
                        'evaluation': eval_data[q_idx]
                    })
            
            # 使用第一个审阅者的评估作为主要评估（因为所有审阅者评分一致）
            merged_evaluation = all_evaluations[0][q_idx].copy()
            
            # 添加所有审阅者的评估信息
            merged_evaluation['all_evaluators'] = all_evaluators_scores
            merged_evaluation['num_evaluators'] = len(all_evaluators_scores)
            merged_evaluation['agreement'] = 'all_agreed'
        else:
            # 选项：只保留一个评估结果（因为评分都一样）
            # 使用第一个审阅者的评估作为代表
            merged_evaluation = all_evaluations[0][q_idx].copy()
            merged_evaluation['num_evaluators'] = len(all_evaluations)
            merged_evaluation['agreement'] = 'all_agreed'
        
        merged_evaluations[q_idx] = merged_evaluation
        
        # 添加问题（如果存在）
        if q_idx < len(all_questions):
            question = all_questions[q_idx].copy()
            question['human_evaluation'] = merged_evaluation
            merged_questions.append(question)
    
    # 计算统计信息
    statistics = {
        'total_evaluators': len(all_evaluations),
        'total_questions': len(common_indices),
        'agreed_questions': len(agreed_indices),
        'agreement_rate': len(agreed_indices) / len(common_indices) if common_indices else 0,
        'threshold': threshold,
        'correlations': correlations,
        'evaluation_files': evaluation_files
    }
    
    result = {
        'statistics': statistics,
        'evaluations': merged_evaluations,
        'questions': merged_questions
    }
    
    # 显示统计信息
    print(f"\n{'='*60}")
    print("合并结果统计")
    print(f"{'='*60}")
    print(f"审阅者数量: {statistics['total_evaluators']}")
    print(f"共同评估问题数: {statistics['total_questions']}")
    print(f"意见一致问题数: {statistics['agreed_questions']}")
    print(f"一致率: {statistics['agreement_rate']*100:.1f}%")
    print(f"{'='*60}\n")
    
    # 保存结果
    if not stats_only and output_file:
        print(f"保存合并结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存成功")
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description='合并多个用户审阅者的评估结果，只保留意见一致的评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 合并多个评估文件
  python merge_human_evaluations.py --files eval1.json eval2.json eval3.json --output merged.json
  
  # 允许评分差异为 1（更宽松的一致性要求）
  python merge_human_evaluations.py --files eval1.json eval2.json --threshold 1 --output merged.json
  
  # 只显示统计信息，不保存结果
  python merge_human_evaluations.py --files eval1.json eval2.json --stats-only
        """
    )
    
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        required=True,
        help='评估文件路径列表（至少 2 个）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件路径（默认：merged_evaluations.json）'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0,
        help='允许的评分差异阈值（默认：0，表示必须完全一致）'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='只显示统计信息，不保存结果'
    )
    
    parser.add_argument(
        '--keep-all-evaluators',
        action='store_true',
        help='保留所有审阅者的评估（即使评分一致，评估文本可能不同）。默认：只保留一个评估结果'
    )
    
    args = parser.parse_args()
    
    if len(args.files) < 2:
        print("错误: 需要至少 2 个评估文件")
        return
    
    # 检查文件是否存在
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"错误: 文件不存在: {file_path}")
            return
    
    # 设置默认输出文件
    if not args.output and not args.stats_only:
        args.output = 'merged_evaluations.json'
    
    # 合并评估
    merge_evaluations(
        args.files,
        threshold=args.threshold,
        output_file=args.output,
        stats_only=args.stats_only,
        keep_all_evaluators=args.keep_all_evaluators
    )

if __name__ == '__main__':
    main()

