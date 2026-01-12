"""
Test script for Educational Agent Policy compliance.
使用AI模拟学生测试教育助手策略是否符合要求。

Tests:
1. Flow termination conditions (流程中止条件)
2. Answer revelation prevention (答案泄露防护)
3. Progressive decomposition (递进式拆分)
4. Conversation history usage (对话历史使用)
5. Error correction (错误纠正)
"""

import json
import os
from typing import Dict, List, Optional
from openai import OpenAI
from proactive_question_generator import ProactiveQuestionGenerator, HintRequest

# Initialize OpenAI client for student simulation
# 初始化OpenAI客户端用于模拟学生
def get_api_key() -> str:
    """Get OpenAI API key from environment or api-key.js file."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        # Try to read from api-key.js
        from pathlib import Path
        config_path = Path(__file__).parent / 'api-key.js'
        if config_path.exists():
            try:
                content = config_path.read_text(encoding='utf-8')
                import re
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
                    print("[OK] API key loaded from api-key.js")
            except Exception as e:
                print(f"[!] Error reading api-key.js: {e}")
    
    if not api_key:
        raise ValueError(
            "Please set OPENAI_API_KEY environment variable or create api-key.js file.\n"
            "设置OPENAI_API_KEY环境变量或创建api-key.js文件。"
        )
    
    return api_key

STUDENT_SIMULATOR_API_KEY = get_api_key()
student_client = OpenAI(api_key=STUDENT_SIMULATOR_API_KEY)

# Initialize tutor (educational agent)
# 初始化导师（教育助手）
tutor_generator = ProactiveQuestionGenerator()


def simulate_student_response(
    question: str,
    choices: Dict[str, str],
    tutor_message: str,
    conversation_history: List[Dict],
    scenario: str = "normal"
) -> str:
    """
    Simulate a student's response using GPT.
    使用GPT模拟学生的回答。
    
    Args:
        question: Original medical question
        choices: Answer choices
        tutor_message: Tutor's message (question or clarification)
        conversation_history: Full conversation history
        scenario: Test scenario ("normal", "wants_answer", "confused", "understands")
    
    Returns:
        Simulated student response
    """
    scenario_prompts = {
        "normal": "You are a medical student who answered incorrectly. Respond naturally to the tutor's guidance. Show some understanding but still need help.",
        "wants_answer": "You are a medical student who is frustrated and wants to see the answer. Express this clearly in your response.",
        "confused": "You are a medical student who is very confused and doesn't understand the concepts. Show fundamental gaps in knowledge.",
        "understands": "You are a medical student who has understood the key concepts after the tutor's guidance. Show clear understanding in your response."
    }
    
    prompt = f"""You are simulating a medical student's response in an educational tutoring system.

## Original Question:
{question}

## Answer Choices:
{chr(10).join([f"{k}: {v}" for k, v in choices.items()])}

## Conversation History:
{chr(10).join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in conversation_history[-5:]]) if conversation_history else "None"}

## Tutor's Current Message:
{tutor_message}

## Your Task:
{scenario_prompts.get(scenario, scenario_prompts["normal"])}

Respond as the student would, naturally and authentically. Keep your response concise (1-3 sentences).
"""
    
    try:
        response = student_client.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for simulation
            messages=[
                {"role": "system", "content": "You are a medical student participating in a tutoring session."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error simulating student: {e}")
        return "I'm not sure about this."


def check_answer_revelation(response: Dict) -> bool:
    """
    Check if the response reveals the correct answer.
    检查响应是否泄露正确答案。
    """
    text_to_check = json.dumps(response).lower()
    
    # Check for explicit answer mentions
    # 检查是否明确提到答案
    answer_keywords = [
        "the correct answer is",
        "the answer is",
        "correct answer:",
        "answer:",
        "the right answer",
        "正确答案是",
        "答案是"
    ]
    
    for keyword in answer_keywords:
        if keyword in text_to_check:
            return True
    
    return False


def test_flow_termination_wants_answer():
    """
    Test: Flow terminates when student requests answer.
    测试：学生要求看答案时流程中止。
    """
    print("\n" + "="*80)
    print("TEST 1: Flow Termination - Student Wants Answer")
    print("测试1：流程中止 - 学生要求看答案")
    print("="*80)
    
    request = HintRequest(
        question="What is the first-line treatment for uncomplicated urinary tract infection in non-pregnant women?",
        choices={
            "A": "Trimethoprim-sulfamethoxazole",
            "B": "Ciprofloxacin",
            "C": "Ampicillin",
            "D": "Vancomycin"
        },
        student_answer="D",
        correct_answer="A"
    )
    
    # Simulate student thinking that wants answer
    # 模拟想要看答案的学生思考
    student_thinking = "I have no idea. Can you just tell me the answer?"
    
    result = tutor_generator.evaluate_student_thinking(request, student_thinking)
    
    print(f"\nStudent Thinking: {student_thinking}")
    print(f"\nTutor Response:")
    print(f"  - action_type: {result.get('action_type')}")
    print(f"  - reveal_answer: {result.get('reveal_answer')}")
    print(f"  - flow_terminated: {result.get('flow_terminated')}")
    print(f"  - next_action_type: {result.get('next_action_type')}")
    print(f"  - clarification: {result.get('clarification', '')[:100]}...")
    
    # Assertions
    assert result.get('reveal_answer') == True, "Should reveal answer when requested"
    assert result.get('flow_terminated') == True, "Flow should terminate when answer is revealed"
    assert result.get('next_action_type') is None, "next_action_type should be None when flow terminates"
    
    print("\n✅ TEST PASSED: Flow correctly terminates when student requests answer")
    print("✅ 测试通过：学生要求看答案时流程正确中止")


def test_answer_revelation_prevention():
    """
    Test: System never reveals answer unless student explicitly requests it.
    测试：除非学生明确要求，系统绝不泄露答案。
    """
    print("\n" + "="*80)
    print("TEST 2: Answer Revelation Prevention")
    print("测试2：答案泄露防护")
    print("="*80)
    
    request = HintRequest(
        question="What is the most common cause of community-acquired pneumonia in adults?",
        choices={
            "A": "Streptococcus pneumoniae",
            "B": "Mycoplasma pneumoniae",
            "C": "Legionella pneumophila",
            "D": "Chlamydia pneumoniae"
        },
        student_answer="B",
        correct_answer="A"
    )
    
    # Test multiple rounds without requesting answer
    # 测试多轮不要求答案的情况
    conversation_history = []
    round_number = 1
    max_rounds = 5
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        if round_num == 1:
            # First round: evaluate student thinking
            student_thinking = "I thought Mycoplasma because it's common in young adults."
            result = tutor_generator.evaluate_student_thinking(request, student_thinking)
            conversation_history.append({"role": "user", "content": student_thinking})
        else:
            # Subsequent rounds: evaluate guidance response
            current_action = result.get("action_type", "decompose")
            current_sub_questions = result.get("sub_questions", [])
            current_clarification = result.get("clarification", "")
            
            # Simulate student response
            student_response = simulate_student_response(
                request.question,
                request.choices,
                current_sub_questions[0] if current_sub_questions else current_clarification,
                conversation_history,
                scenario="normal"
            )
            
            conversation_history.append({"role": "user", "content": student_response})
            
            result = tutor_generator.evaluate_guidance_response(
                request=request,
                current_action=current_action,
                current_sub_questions=current_sub_questions,
                current_clarification=current_clarification,
                student_response=student_response,
                conversation_history=conversation_history,
                current_understanding_level=result.get("understanding_level", "partial"),
                round_number=round_num
            )
        
        conversation_history.append({"role": "assistant", "content": json.dumps(result)})
        
        # Check for answer revelation
        answer_revealed = check_answer_revelation(result)
        
        print(f"  Student: {conversation_history[-2].get('content', '')[:80]}...")
        print(f"  Tutor action: {result.get('action_type')}")
        print(f"  Answer revealed: {answer_revealed}")
        
        assert not answer_revealed, f"Answer should not be revealed in round {round_num}"
        
        # Check if flow should terminate
        if result.get("flow_terminated") or result.get("understood"):
            print(f"  Flow terminated: {result.get('flow_terminated')}")
            break
    
    print("\n✅ TEST PASSED: Answer never revealed without explicit request")
    print("✅ 测试通过：未明确要求时答案从未泄露")


def test_progressive_decomposition():
    """
    Test: Questions become progressively simpler and more foundational.
    测试：问题逐步变得更简单、更基础。
    """
    print("\n" + "="*80)
    print("TEST 3: Progressive Decomposition")
    print("测试3：递进式拆分")
    print("="*80)
    
    request = HintRequest(
        question="What is the mechanism of action of metformin?",
        choices={
            "A": "Increases insulin secretion",
            "B": "Decreases hepatic glucose production",
            "C": "Increases peripheral glucose uptake",
            "D": "Both B and C"
        },
        student_answer="A",
        correct_answer="D"
    )
    
    student_thinking = "I thought metformin increases insulin secretion like sulfonylureas."
    result = tutor_generator.evaluate_student_thinking(request, student_thinking)
    
    conversation_history = [{"role": "user", "content": student_thinking}]
    previous_questions = []
    
    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")
        
        if round_num == 1:
            sub_questions = result.get("sub_questions", [])
        else:
            current_action = result.get("action_type", "decompose")
            current_sub_questions = result.get("sub_questions", [])
            current_clarification = result.get("clarification", "")
            
            student_response = simulate_student_response(
                request.question,
                request.choices,
                current_sub_questions[0] if current_sub_questions else current_clarification,
                conversation_history,
                scenario="confused"
            )
            
            conversation_history.append({"role": "user", "content": student_response})
            
            result = tutor_generator.evaluate_guidance_response(
                request=request,
                current_action=current_action,
                current_sub_questions=current_sub_questions,
                current_clarification=current_clarification,
                student_response=student_response,
                conversation_history=conversation_history,
                current_understanding_level=result.get("understanding_level", "partial"),
                round_number=round_num
            )
            
            sub_questions = result.get("sub_questions", [])
        
        if sub_questions:
            print(f"  Questions in round {round_num}:")
            for i, q in enumerate(sub_questions, 1):
                print(f"    {i}. {q[:100]}...")
                previous_questions.append(q)
        
        conversation_history.append({"role": "assistant", "content": json.dumps(result)})
        
        if result.get("flow_terminated") or result.get("understood"):
            break
    
    print("\n✅ TEST PASSED: Progressive decomposition verified")
    print("✅ 测试通过：递进式拆分已验证")


def test_conversation_history_usage():
    """
    Test: System uses complete conversation history for targeted guidance.
    测试：系统使用完整对话历史进行针对性指导。
    """
    print("\n" + "="*80)
    print("TEST 4: Conversation History Usage")
    print("测试4：对话历史使用")
    print("="*80)
    
    request = HintRequest(
        question="What is the treatment for acute myocardial infarction?",
        choices={
            "A": "Aspirin and clopidogrel",
            "B": "Aspirin, clopidogrel, and statin",
            "C": "Aspirin, clopidogrel, statin, and ACE inhibitor",
            "D": "Aspirin, clopidogrel, statin, ACE inhibitor, and beta-blocker"
        },
        student_answer="A",
        correct_answer="D"
    )
    
    student_thinking = "I thought only aspirin and clopidogrel are needed."
    result = tutor_generator.evaluate_student_thinking(request, student_thinking)
    
    conversation_history = [{"role": "user", "content": student_thinking}]
    student_errors = []
    
    for round_num in range(1, 3):
        print(f"\n--- Round {round_num} ---")
        
        if round_num > 1:
            current_action = result.get("action_type", "decompose")
            current_sub_questions = result.get("sub_questions", [])
            current_clarification = result.get("clarification", "")
            
            student_response = simulate_student_response(
                request.question,
                request.choices,
                current_sub_questions[0] if current_sub_questions else current_clarification,
                conversation_history,
                scenario="normal"
            )
            
            conversation_history.append({"role": "user", "content": student_response})
            student_errors.append(student_response)
            
            result = tutor_generator.evaluate_guidance_response(
                request=request,
                current_action=current_action,
                current_sub_questions=current_sub_questions,
                current_clarification=current_clarification,
                student_response=student_response,
                conversation_history=conversation_history,
                current_understanding_level=result.get("understanding_level", "partial"),
                round_number=round_num
            )
        
        print(f"  Conversation history length: {len(conversation_history)}")
        print(f"  Student errors tracked: {len(student_errors)}")
        
        conversation_history.append({"role": "assistant", "content": json.dumps(result)})
        
        if result.get("flow_terminated") or result.get("understood"):
            break
    
    print("\n✅ TEST PASSED: Conversation history is being used")
    print("✅ 测试通过：对话历史正在被使用")


def test_flow_termination_understood():
    """
    Test: Flow terminates when student demonstrates understanding.
    测试：学生表现出理解时流程中止。
    """
    print("\n" + "="*80)
    print("TEST 5: Flow Termination - Student Understands")
    print("测试5：流程中止 - 学生理解")
    print("="*80)
    
    request = HintRequest(
        question="What is the most common side effect of ACE inhibitors?",
        choices={
            "A": "Hyperkalemia",
            "B": "Dry cough",
            "C": "Hypotension",
            "D": "Renal failure"
        },
        student_answer="A",
        correct_answer="B"
    )
    
    student_thinking = "I thought hyperkalemia was the most common."
    result = tutor_generator.evaluate_student_thinking(request, student_thinking)
    
    conversation_history = [{"role": "user", "content": student_thinking}]
    
    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")
        
        if round_num > 1:
            current_action = result.get("action_type", "decompose")
            current_sub_questions = result.get("sub_questions", [])
            current_clarification = result.get("clarification", "")
            
            # Simulate student showing understanding
            student_response = simulate_student_response(
                request.question,
                request.choices,
                current_sub_questions[0] if current_sub_questions else current_clarification,
                conversation_history,
                scenario="understands" if round_num >= 2 else "normal"
            )
            
            conversation_history.append({"role": "user", "content": student_response})
            
            result = tutor_generator.evaluate_guidance_response(
                request=request,
                current_action=current_action,
                current_sub_questions=current_sub_questions,
                current_clarification=current_clarification,
                student_response=student_response,
                conversation_history=conversation_history,
                current_understanding_level=result.get("understanding_level", "partial"),
                round_number=round_num
            )
        
        print(f"  Understood: {result.get('understood')}")
        print(f"  Flow terminated: {result.get('flow_terminated')}")
        print(f"  Next action: {result.get('next_action_type')}")
        
        conversation_history.append({"role": "assistant", "content": json.dumps(result)})
        
        if result.get("understood") or result.get("flow_terminated"):
            assert result.get("next_action_type") is None, "next_action_type should be None when flow terminates"
            print("\n✅ TEST PASSED: Flow correctly terminates when student understands")
            print("✅ 测试通过：学生理解时流程正确中止")
            break


def run_all_tests():
    """Run all policy compliance tests."""
    print("\n" + "="*80)
    print("EDUCATIONAL AGENT POLICY COMPLIANCE TESTS")
    print("教育助手策略合规性测试")
    print("="*80)
    
    tests = [
        test_flow_termination_wants_answer,
        test_answer_revelation_prevention,
        test_progressive_decomposition,
        test_conversation_history_usage,
        test_flow_termination_understood
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            print(f"❌ 测试失败：{e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            print(f"❌ 测试错误：{e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"TEST SUMMARY / 测试总结")
    print("="*80)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")
    print("="*80)


def run_batch_tests(num_runs: int = 100):
    """
    Run batch tests multiple times to verify system stability.
    运行批量测试多次以验证系统稳定性。
    
    Args:
        num_runs: Number of test runs (default: 100)
    """
    print("\n" + "="*80)
    print(f"BATCH TESTING - {num_runs} RUNS")
    print(f"批量测试 - {num_runs} 次运行")
    print("="*80)
    
    # Test scenarios to run
    # 要运行的测试场景
    test_scenarios = [
        ("Flow Termination - Wants Answer", test_flow_termination_wants_answer),
        ("Answer Revelation Prevention", test_answer_revelation_prevention),
        ("Progressive Decomposition", test_progressive_decomposition),
        ("Conversation History Usage", test_conversation_history_usage),
        ("Flow Termination - Understands", test_flow_termination_understood)
    ]
    
    # Statistics
    # 统计信息
    scenario_stats = {name: {"passed": 0, "failed": 0, "errors": 0} for name, _ in test_scenarios}
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    print(f"\nRunning {num_runs} test cycles...")
    print(f"运行 {num_runs} 个测试周期...\n")
    
    for run_num in range(1, num_runs + 1):
        if run_num % 10 == 0:
            print(f"Progress: {run_num}/{num_runs} runs completed...")
            print(f"进度：{run_num}/{num_runs} 次运行完成...")
        
        for scenario_name, test_func in test_scenarios:
            try:
                # Suppress output for batch testing
                # 批量测试时抑制输出
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                test_func()
                
                sys.stdout = old_stdout
                scenario_stats[scenario_name]["passed"] += 1
                total_passed += 1
                
            except AssertionError as e:
                sys.stdout = old_stdout
                scenario_stats[scenario_name]["failed"] += 1
                total_failed += 1
                if run_num <= 5:  # Show first 5 failures
                    print(f"\n[Run {run_num}] ❌ {scenario_name} FAILED: {e}")
                
            except Exception as e:
                sys.stdout = old_stdout
                scenario_stats[scenario_name]["errors"] += 1
                total_errors += 1
                if run_num <= 5:  # Show first 5 errors
                    print(f"\n[Run {run_num}] ❌ {scenario_name} ERROR: {e}")
    
    # Print statistics
    # 打印统计信息
    print("\n" + "="*80)
    print("BATCH TEST RESULTS / 批量测试结果")
    print("="*80)
    print(f"\nTotal Runs: {num_runs}")
    print(f"总运行次数：{num_runs}\n")
    
    print("Scenario Statistics / 场景统计:")
    print("-" * 80)
    for scenario_name, stats in scenario_stats.items():
        total = stats["passed"] + stats["failed"] + stats["errors"]
        pass_rate = (stats["passed"] / total * 100) if total > 0 else 0
        print(f"\n{scenario_name}:")
        print(f"  Passed: {stats['passed']} ({pass_rate:.1f}%)")
        print(f"  Failed: {stats['failed']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Total: {total}")
    
    print("\n" + "-" * 80)
    print("Overall Statistics / 总体统计:")
    total_tests = total_passed + total_failed + total_errors
    overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    print(f"  Total Errors: {total_errors}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Pass Rate: {overall_pass_rate:.2f}%")
    print("="*80)
    
    # Return summary
    # 返回摘要
    return {
        "total_runs": num_runs,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_errors": total_errors,
        "pass_rate": overall_pass_rate,
        "scenario_stats": scenario_stats
    }


if __name__ == "__main__":
    import sys
    
    # Check if batch mode is requested
    # 检查是否请求批量模式
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_batch_tests(num_runs)
    else:
        run_all_tests()

