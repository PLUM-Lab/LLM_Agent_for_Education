"""
================================================================================
知识追踪系统测试脚本
Knowledge Tracking System Test Script
================================================================================

功能：
    - 批量测试 generated_domain_questions.json 中的477个问题
    - 模拟学生答题过程（可设置答对率）
    - 生成知识追踪报告
    - 分析知识覆盖情况

使用方法：
    python test_knowledge_tracking.py [选项]

选项：
    --correct-rate: 答对率（0.0-1.0），默认0.7（70%答对）
    --student-name: 学生姓名，默认"TestStudent"
    --max-questions: 最大测试问题数，默认全部（477）
    --random: 随机顺序测试
    --report: 生成详细报告文件
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# 模拟localStorage的简单实现
class MockLocalStorage:
    """模拟浏览器的localStorage功能"""
    def __init__(self):
        self.storage = {}
    
    def getItem(self, key: str) -> str:
        return self.storage.get(key)
    
    def setItem(self, key: str, value: str):
        self.storage[key] = value
    
    def removeItem(self, key: str):
        if key in self.storage:
            del self.storage[key]

# 全局模拟localStorage
mock_storage = MockLocalStorage()

def initialize_student_profile(user_name: str) -> Dict:
    """初始化学生档案"""
    return {
        "userName": user_name,
        "questionsAnswered": 0,
        "knowledgeMap": {},
        "createdAt": datetime.now().isoformat(),
        "lastUpdated": datetime.now().isoformat()
    }

def get_knowledge_component_key(question: Dict) -> str:
    """从问题中获取知识组件键"""
    if not question:
        return None
    
    if question.get("domain") and question.get("topic") and question.get("subtopic"):
        return f"{question['domain']}->{question['topic']}->{question['subtopic']}"
    
    return question.get("id", f"unknown_{random.randint(1000, 9999)}")

def update_knowledge_status(profile: Dict, question: Dict, is_correct: bool):
    """更新知识状态"""
    key = get_knowledge_component_key(question)
    if not key:
        return
    
    if key not in profile["knowledgeMap"]:
        profile["knowledgeMap"][key] = {
            "status": "unknown",
            "lastUpdated": datetime.now().isoformat(),
            "questionsAttempted": 0,
            "questionsCorrect": 0,
            "domain": question.get("domain", "unknown"),
            "topic": question.get("topic", "unknown"),
            "subtopic": question.get("subtopic", "unknown")
        }
    
    component = profile["knowledgeMap"][key]
    component["questionsAttempted"] += 1
    
    if is_correct:
        component["status"] = "known"
        component["questionsCorrect"] += 1
    else:
        component["status"] = "unknown"
    
    component["lastUpdated"] = datetime.now().isoformat()
    profile["questionsAnswered"] += 1
    profile["lastUpdated"] = datetime.now().isoformat()

def simulate_answer(question: Dict, correct_rate: float) -> Tuple[bool, str]:
    """模拟学生答题"""
    correct_answer = question.get("correct_answer", "A")
    options = list(question.get("options", {}).keys())
    
    # 根据答对率决定是否答对
    if random.random() < correct_rate:
        # 答对
        return True, correct_answer
    else:
        # 答错：随机选择一个错误答案
        wrong_options = [opt for opt in options if opt != correct_answer]
        wrong_answer = random.choice(wrong_options) if wrong_options else options[0]
        return False, wrong_answer

def test_questions(
    questions: List[Dict],
    student_name: str = "TestStudent",
    correct_rate: float = 0.7,
    max_questions: int = None,
    random_order: bool = False
) -> Dict:
    """测试问题并追踪知识状态"""
    # 初始化学生档案
    profile = initialize_student_profile(student_name)
    
    # 限制问题数量
    test_questions = questions[:max_questions] if max_questions else questions
    
    # 随机顺序
    if random_order:
        random.shuffle(test_questions)
    
    # 统计信息
    stats = {
        "total_questions": len(test_questions),
        "correct_answers": 0,
        "wrong_answers": 0,
        "by_domain": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0}),
        "by_topic": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0}),
        "by_subtopic": defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})
    }
    
    print(f"\n{'='*60}")
    print(f"开始测试 {len(test_questions)} 个问题")
    print(f"学生姓名: {student_name}")
    print(f"答对率设置: {correct_rate*100:.1f}%")
    print(f"{'='*60}\n")
    
    # 逐个测试问题
    for i, question in enumerate(test_questions, 1):
        is_correct, answer = simulate_answer(question, correct_rate)
        
        # 更新知识状态
        update_knowledge_status(profile, question, is_correct)
        
        # 更新统计
        if is_correct:
            stats["correct_answers"] += 1
        else:
            stats["wrong_answers"] += 1
        
        # 按领域/主题/子主题统计
        domain = question.get("domain", "unknown")
        topic = question.get("topic", "unknown")
        subtopic = question.get("subtopic", "unknown")
        
        stats["by_domain"][domain]["total"] += 1
        stats["by_topic"][f"{domain}->{topic}"]["total"] += 1
        stats["by_subtopic"][f"{domain}->{topic}->{subtopic}"]["total"] += 1
        
        if is_correct:
            stats["by_domain"][domain]["correct"] += 1
            stats["by_topic"][f"{domain}->{topic}"]["correct"] += 1
            stats["by_subtopic"][f"{domain}->{topic}->{subtopic}"]["correct"] += 1
        else:
            stats["by_domain"][domain]["wrong"] += 1
            stats["by_topic"][f"{domain}->{topic}"]["wrong"] += 1
            stats["by_subtopic"][f"{domain}->{topic}->{subtopic}"]["wrong"] += 1
        
        # 进度显示
        if i % 50 == 0 or i == len(test_questions):
            print(f"进度: {i}/{len(test_questions)} ({i/len(test_questions)*100:.1f}%)")
    
    return profile, stats

def generate_report(profile: Dict, stats: Dict, output_file: str = None):
    """生成知识追踪报告"""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("知识追踪系统测试报告")
    report_lines.append("Knowledge Tracking System Test Report")
    report_lines.append("="*80)
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"学生姓名: {profile['userName']}")
    report_lines.append(f"测试时间: {profile['createdAt']} - {profile['lastUpdated']}")
    
    # 总体统计
    report_lines.append("\n" + "-"*80)
    report_lines.append("总体统计 / Overall Statistics")
    report_lines.append("-"*80)
    report_lines.append(f"已回答问题数: {profile['questionsAnswered']}")
    report_lines.append(f"答对题数: {stats['correct_answers']}")
    report_lines.append(f"答错题数: {stats['wrong_answers']}")
    actual_rate = stats['correct_answers'] / stats['total_questions'] * 100 if stats['total_questions'] > 0 else 0
    report_lines.append(f"实际答对率: {actual_rate:.2f}%")
    
    # 知识组件统计
    knowledge_map = profile.get("knowledgeMap", {})
    total_components = len(knowledge_map)
    known_components = sum(1 for c in knowledge_map.values() if c.get("status") == "known")
    unknown_components = total_components - known_components
    
    report_lines.append("\n" + "-"*80)
    report_lines.append("知识组件统计 / Knowledge Components Statistics")
    report_lines.append("-"*80)
    report_lines.append(f"总知识组件数: {total_components}")
    report_lines.append(f"已知组件数: {known_components} ({known_components/total_components*100:.1f}%)" if total_components > 0 else "已知组件数: 0")
    report_lines.append(f"未知组件数: {unknown_components} ({unknown_components/total_components*100:.1f}%)" if total_components > 0 else "未知组件数: 0")
    
    # 按领域统计
    report_lines.append("\n" + "-"*80)
    report_lines.append("按领域统计 / Statistics by Domain")
    report_lines.append("-"*80)
    for domain, domain_stats in sorted(stats["by_domain"].items()):
        total = domain_stats["total"]
        correct = domain_stats["correct"]
        rate = correct / total * 100 if total > 0 else 0
        report_lines.append(f"{domain}: {correct}/{total} ({rate:.1f}%)")
    
    # 知识组件详情（前20个）
    report_lines.append("\n" + "-"*80)
    report_lines.append("知识组件详情（最近20个）/ Knowledge Components Details (Top 20)")
    report_lines.append("-"*80)
    
    sorted_components = sorted(
        knowledge_map.items(),
        key=lambda x: x[1].get("lastUpdated", ""),
        reverse=True
    )[:20]
    
    for key, component in sorted_components:
        status = component.get("status", "unknown")
        status_icon = "✓" if status == "known" else "✗"
        attempted = component.get("questionsAttempted", 0)
        correct = component.get("questionsCorrect", 0)
        accuracy = correct / attempted * 100 if attempted > 0 else 0
        
        report_lines.append(f"\n{status_icon} {component.get('subtopic', 'Unknown')}")
        report_lines.append(f"   领域: {component.get('domain', 'Unknown')} → 主题: {component.get('topic', 'Unknown')}")
        report_lines.append(f"   准确率: {accuracy:.1f}% ({correct}/{attempted})")
        report_lines.append(f"   状态: {'已知' if status == 'known' else '未知'}")
    
    # 所有知识组件列表
    report_lines.append("\n" + "-"*80)
    report_lines.append("所有知识组件列表 / All Knowledge Components List")
    report_lines.append("-"*80)
    
    for key, component in sorted(knowledge_map.items(), key=lambda x: x[1].get("subtopic", "")):
        status = component.get("status", "unknown")
        status_icon = "✓" if status == "known" else "✗"
        attempted = component.get("questionsAttempted", 0)
        correct = component.get("questionsCorrect", 0)
        accuracy = correct / attempted * 100 if attempted > 0 else 0
        
        report_lines.append(
            f"{status_icon} {component.get('domain', 'Unknown')} → "
            f"{component.get('topic', 'Unknown')} → "
            f"{component.get('subtopic', 'Unknown')} | "
            f"准确率: {accuracy:.1f}% ({correct}/{attempted})"
        )
    
    report_text = "\n".join(report_lines)
    
    # 输出到控制台
    print("\n" + report_text)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存到: {output_file}")
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description='测试知识追踪系统')
    parser.add_argument('--correct-rate', type=float, default=0.7,
                       help='答对率 (0.0-1.0)，默认0.7 (70%%)')
    parser.add_argument('--student-name', type=str, default='TestStudent',
                       help='学生姓名，默认TestStudent')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='最大测试问题数，默认全部')
    parser.add_argument('--random', action='store_true',
                       help='随机顺序测试')
    parser.add_argument('--report', type=str, default=None,
                       help='生成报告文件路径（可选）')
    
    args = parser.parse_args()
    
    # 加载问题
    questions_file = Path('generated_domain_questions.json')
    if not questions_file.exists():
        print(f"错误: 找不到问题文件 {questions_file}")
        return
    
    print(f"加载问题文件: {questions_file}")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"已加载 {len(questions)} 个问题")
    
    # 测试问题
    profile, stats = test_questions(
        questions=questions,
        student_name=args.student_name,
        correct_rate=args.correct_rate,
        max_questions=args.max_questions,
        random_order=args.random
    )
    
    # 生成报告
    report_file = args.report or f"knowledge_tracking_report_{args.student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    generate_report(profile, stats, report_file)
    
    # 保存profile到JSON（可选）
    profile_file = f"student_profile_{args.student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(profile_file, 'w', encoding='utf-8') as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"\n学生档案已保存到: {profile_file}")

if __name__ == '__main__':
    main()

