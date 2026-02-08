"""
================================================================================
完成所有问题后的分析脚本
Analysis Script After Completing All Questions
================================================================================

功能：
    - 分析完成477个问题后的学生档案
    - 生成详细的知识覆盖报告
    - 识别薄弱领域和需要复习的知识组件
    - 提供学习建议

使用方法：
    python analyze_completed_profile.py <profile_json_file>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_profile(file_path: str) -> Dict:
    """加载学生档案JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_profile(profile: Dict) -> Dict:
    """分析学生档案，生成统计信息"""
    knowledge_map = profile.get("knowledgeMap", {})
    
    # 总体统计
    total_components = len(knowledge_map)
    known_components = sum(1 for c in knowledge_map.values() if c.get("status") == "known")
    unknown_components = total_components - known_components
    coverage_rate = (known_components / total_components * 100) if total_components > 0 else 0
    
    # 按领域统计
    by_domain = defaultdict(lambda: {"total": 0, "known": 0, "unknown": 0, "components": []})
    for key, component in knowledge_map.items():
        domain = component.get("domain", "unknown")
        by_domain[domain]["total"] += 1
        by_domain[domain]["components"].append(component)
        if component.get("status") == "known":
            by_domain[domain]["known"] += 1
        else:
            by_domain[domain]["unknown"] += 1
    
    # 按主题统计
    by_topic = defaultdict(lambda: {"total": 0, "known": 0, "unknown": 0})
    for component in knowledge_map.values():
        topic_key = f"{component.get('domain', 'unknown')}->{component.get('topic', 'unknown')}"
        by_topic[topic_key]["total"] += 1
        if component.get("status") == "known":
            by_topic[topic_key]["known"] += 1
        else:
            by_topic[topic_key]["unknown"] += 1
    
    # 识别薄弱领域（答对率<70%）
    weak_domains = []
    for domain, stats in by_domain.items():
        if stats["total"] > 0:
            rate = stats["known"] / stats["total"] * 100
            if rate < 70:
                weak_domains.append({
                    "domain": domain,
                    "rate": rate,
                    "known": stats["known"],
                    "total": stats["total"]
                })
    
    # 识别需要复习的知识组件（答错或准确率<50%）
    need_review = []
    for key, component in knowledge_map.items():
        attempted = component.get("questionsAttempted", 0)
        correct = component.get("questionsCorrect", 0)
        if attempted > 0:
            accuracy = correct / attempted * 100
            if component.get("status") == "unknown" or accuracy < 50:
                need_review.append({
                    "key": key,
                    "component": component,
                    "accuracy": accuracy,
                    "attempted": attempted,
                    "correct": correct
                })
    
    # 按准确率排序
    need_review.sort(key=lambda x: x["accuracy"])
    
    # 识别掌握最好的领域（答对率>90%）
    strong_domains = []
    for domain, stats in by_domain.items():
        if stats["total"] > 0:
            rate = stats["known"] / stats["total"] * 100
            if rate >= 90:
                strong_domains.append({
                    "domain": domain,
                    "rate": rate,
                    "known": stats["known"],
                    "total": stats["total"]
                })
    
    return {
        "total_questions": profile.get("questionsAnswered", 0),
        "total_components": total_components,
        "known_components": known_components,
        "unknown_components": unknown_components,
        "coverage_rate": coverage_rate,
        "by_domain": dict(by_domain),
        "by_topic": dict(by_topic),
        "weak_domains": weak_domains,
        "strong_domains": strong_domains,
        "need_review": need_review[:20],  # 前20个最需要复习的
        "knowledge_map": knowledge_map
    }

def generate_report(analysis: Dict, profile: Dict) -> str:
    """生成分析报告"""
    lines = []
    
    lines.append("="*80)
    lines.append("完成所有问题后的知识追踪分析报告")
    lines.append("Knowledge Tracking Analysis Report After Completing All Questions")
    lines.append("="*80)
    lines.append("")
    
    # 总体情况
    lines.append("-"*80)
    lines.append("📊 总体情况 / Overall Statistics")
    lines.append("-"*80)
    lines.append(f"已回答问题总数: {analysis['total_questions']}")
    lines.append(f"知识组件总数: {analysis['total_components']}")
    lines.append(f"已知组件数: {analysis['known_components']} ({analysis['coverage_rate']:.1f}%)")
    lines.append(f"未知组件数: {analysis['unknown_components']} ({100-analysis['coverage_rate']:.1f}%)")
    lines.append("")
    
    # 按领域统计
    lines.append("-"*80)
    lines.append("📚 按领域统计 / Statistics by Domain")
    lines.append("-"*80)
    for domain, stats in sorted(analysis["by_domain"].items()):
        rate = stats["known"] / stats["total"] * 100 if stats["total"] > 0 else 0
        status_icon = "✅" if rate >= 80 else "⚠️" if rate >= 60 else "❌"
        lines.append(f"{status_icon} {domain}: {stats['known']}/{stats['total']} ({rate:.1f}%)")
    lines.append("")
    
    # 掌握最好的领域
    if analysis["strong_domains"]:
        lines.append("-"*80)
        lines.append("🌟 掌握最好的领域 / Strongest Domains (≥90%)")
        lines.append("-"*80)
        for domain_info in sorted(analysis["strong_domains"], key=lambda x: x["rate"], reverse=True):
            lines.append(f"✅ {domain_info['domain']}: {domain_info['rate']:.1f}% "
                        f"({domain_info['known']}/{domain_info['total']})")
        lines.append("")
    
    # 薄弱领域
    if analysis["weak_domains"]:
        lines.append("-"*80)
        lines.append("⚠️ 薄弱领域 / Weak Domains (<70%)")
        lines.append("-"*80)
        for domain_info in sorted(analysis["weak_domains"], key=lambda x: x["rate"]):
            lines.append(f"❌ {domain_info['domain']}: {domain_info['rate']:.1f}% "
                        f"({domain_info['known']}/{domain_info['total']})")
        lines.append("")
    
    # 需要复习的知识组件
    if analysis["need_review"]:
        lines.append("-"*80)
        lines.append("📝 需要重点复习的知识组件 / Knowledge Components Needing Review")
        lines.append("-"*80)
        lines.append("(按准确率从低到高排序，显示前20个)")
        lines.append("")
        for i, item in enumerate(analysis["need_review"], 1):
            comp = item["component"]
            lines.append(f"{i}. {comp.get('subtopic', 'Unknown')}")
            lines.append(f"   领域: {comp.get('domain', 'Unknown')} → 主题: {comp.get('topic', 'Unknown')}")
            lines.append(f"   准确率: {item['accuracy']:.1f}% ({item['correct']}/{item['attempted']})")
            lines.append(f"   状态: {'已知' if comp.get('status') == 'known' else '未知'}")
            lines.append("")
    
    # 学习建议
    lines.append("-"*80)
    lines.append("💡 学习建议 / Learning Recommendations")
    lines.append("-"*80)
    
    if analysis["coverage_rate"] >= 80:
        lines.append("🎉 恭喜！你已经掌握了大部分知识组件。")
        lines.append("建议：")
        lines.append("  1. 重点复习薄弱领域的知识组件")
        lines.append("  2. 定期回顾已掌握的知识，防止遗忘")
        lines.append("  3. 可以开始更高级的学习内容")
    elif analysis["coverage_rate"] >= 60:
        lines.append("👍 你已经掌握了相当一部分知识。")
        lines.append("建议：")
        lines.append("  1. 系统复习薄弱领域")
        lines.append("  2. 针对准确率<50%的知识组件进行专项练习")
        lines.append("  3. 巩固已掌握的知识")
    else:
        lines.append("📚 还有较大的提升空间。")
        lines.append("建议：")
        lines.append("  1. 重点学习薄弱领域的基础知识")
        lines.append("  2. 多做相关题目的练习")
        lines.append("  3. 寻求老师或同学的帮助")
    
    lines.append("")
    
    # 下一步行动
    lines.append("-"*80)
    lines.append("🚀 下一步行动 / Next Steps")
    lines.append("-"*80)
    lines.append("1. 针对薄弱领域进行专项复习")
    lines.append("2. 重新练习准确率<50%的知识组件相关题目")
    lines.append("3. 定期回顾，防止知识遗忘")
    lines.append("4. 可以开始新的学习内容或更高级的题目")
    lines.append("")
    
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_completed_profile.py <profile_json_file>")
        print("示例: python analyze_completed_profile.py student_profile_FullTest_*.json")
        sys.exit(1)
    
    profile_file = sys.argv[1]
    
    if not Path(profile_file).exists():
        print(f"错误: 找不到文件 {profile_file}")
        sys.exit(1)
    
    print(f"加载学生档案: {profile_file}")
    profile = load_profile(profile_file)
    
    print("分析中...")
    analysis = analyze_profile(profile)
    
    print("生成报告...")
    report = generate_report(analysis, profile)
    
    # 输出报告
    print("\n" + report)
    
    # 保存报告
    output_file = f"analysis_report_{Path(profile_file).stem}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {output_file}")

if __name__ == '__main__':
    main()

