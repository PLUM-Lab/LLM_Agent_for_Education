"""
================================================================================
Proactive Question Generator - Socratic Tutoring System
主动式问题生成器 - 苏格拉底式教学系统
================================================================================

Overview:
    Generates hint sub-questions to guide students towards the correct answer
    using Socratic questioning methodology. Based on the paper:
    "The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models"
    (arXiv:2305.14999)
    
    概述：
    使用苏格拉底式提问方法生成提示性子问题，引导学生找到正确答案。
    基于论文："The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models"

Approach:
    The Socratic method decomposes complex problems into simpler sub-questions,
    guiding students to discover the answer themselves rather than being told.
    
    方法：
    苏格拉底式方法将复杂问题分解为更简单的子问题，引导学生自己发现答案，
    而不是直接告诉他们答案。
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Socratic Questioning Flow                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Student Wrong Answer                                                   │
    │        │                                                                 │
    │        ▼                                                                 │
    │   ┌─────────────────────┐                                               │
    │   │  Analyze Mistake    │  Identify knowledge gap or misconception       │
    │   └─────────────────────┘                                               │
    │        │                                                                 │
    │        ▼                                                                 │
    │   ┌─────────────────────┐                                               │
    │   │  Generate Hints     │  Create 2-3 guiding sub-questions              │
    │   │  (Sub-questions)    │  From basic concepts to specific application   │
    │   └─────────────────────┘                                               │
    │        │                                                                 │
    │        ▼                                                                 │
    │   ┌─────────────────────┐                                               │
    │   │  Multi-Round Hints  │  If still stuck, provide progressively         │
    │   │  (If needed)        │  more specific guidance                        │
    │   └─────────────────────┘                                               │
    │        │                                                                 │
    │        ▼                                                                 │
    │   Student Discovers Answer                                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Input:
    - question: The original medical question (原始医学问题)
    - choices: Dictionary of answer options (A, B, C, D, E) (答案选项字典)
    - student_answer: The wrong answer selected by student (学生选择的错误答案)
    - correct_answer: The correct answer key (A, B, C, D, E) (正确答案)
    - conversation_history: Previous hint rounds (for multi-round guidance) (对话历史，用于多轮引导)

Output:
    - knowledge_blocks: List of sub-questions WITH their answers (knowledge building blocks)
      知识块：包含答案的子问题列表（知识构建块）
    - explanation: Brief context for the hints (without revealing answer)
      解释：提示的简要上下文（不直接揭示答案）

Trigger Conditions:
    触发条件：
    1. Student provides wrong prediction (学生提供错误答案)
    2. Student explicitly asks "provide some hints" or similar (学生明确要求提示)

Author: AI for Education
================================================================================
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    import subprocess
    subprocess.check_call(["pip", "install", "openai"])
    from openai import OpenAI


# =============================================================================
# Data Models
# 数据模型
# =============================================================================

class HintLevel(Enum):
    """Hint difficulty levels - progressively more specific
    提示难度级别 - 逐步更加具体"""
    CONCEPTUAL = "conceptual"      # Round 1: Basic concept questions (第1轮：基础概念问题)
    ANALYTICAL = "analytical"      # Round 2: Analyze the options (第2轮：分析选项)
    DIRECTIVE = "directive"        # Round 3: Direct towards answer (第3轮：直接指向答案)


@dataclass
class HintRequest:
    """Request for generating hints
    生成提示的请求对象"""
    question: str  # 原始医学问题
    choices: Dict[str, str]  # {"A": "option text", "B": "...", ...} 答案选项字典
    student_answer: str      # e.g., "B" 学生选择的答案，例如 "B"
    correct_answer: str      # e.g., "A" 正确答案，例如 "A"
    explanations: Optional[Dict[str, str]] = None  # Explanations for each option (每个选项的解释)
    conversation_history: Optional[List[Dict]] = None  # Previous rounds (之前的对话轮次)
    source_context: Optional[str] = None  # RAG retrieved context (RAG检索到的上下文)
    

@dataclass
class ReasoningStep:
    """A single step in the clinical reasoning chain (MedTutor-R1 style)
    临床推理链中的单个步骤（MedTutor-R1风格）"""
    step_id: str                         # Step ID (e.g., "1", "1.1", "1.2") 步骤ID
    key_question: str                    # The question the student must think about 学生需要思考的关键问题
    step_summary: str                    # Why this step matters in the reasoning chain 此步骤在推理链中的重要性
    expected_understanding: str          # What the student should realize 学生应该理解的内容
    student_response: str = ""           # Student's response (filled during interaction) 学生的回答（交互时填充）
    is_understood: Optional[bool] = None # Whether student showed understanding 学生是否表现出理解
    feedback: str = ""                   # Feedback on response 对回答的反馈
    sub_steps: List['ReasoningStep'] = field(default_factory=list)  # If struggled, simpler sub-steps 如果遇到困难，更简单的子步骤


@dataclass
class ProblemDecomposition:
    """Complete decomposition of a medical problem (MedTutor-R1 style)
    医学问题的完整分解（MedTutor-R1风格）"""
    original_question: str               # The original question 原始问题
    reasoning_steps: List[ReasoningStep] # The reasoning chain 推理步骤链
    synthesis_step: str                  # Final synthesis to reach the answer 最终综合得出答案
    current_step_index: int = 0          # Which step we're on 当前所在的步骤索引


@dataclass
class SocraticResponse:
    """Response containing the problem decomposition
    包含问题分解的响应对象"""
    decomposition: ProblemDecomposition  # The reasoning steps 问题分解（推理步骤）
    active_step_id: str                  # Current step being worked on 当前正在处理的步骤ID
    total_steps: int                     # Total number of steps 总步骤数
    is_complete: bool                    # Whether all steps are done 是否所有步骤都已完成


@dataclass
class EvaluationResult:
    """Result of evaluating student's response to a reasoning step
    评估学生对推理步骤回答的结果"""
    step_id: str                         # Which step was answered 被回答的步骤ID
    understood: bool                     # Whether student showed understanding 学生是否表现出理解
    feedback: str                        # Encouraging feedback 鼓励性反馈
    sub_steps: List[ReasoningStep] = field(default_factory=list)  # If not understood, simpler sub-steps 如果不理解，更简单的子步骤
    missing_concept: str = ""            # What concept they need to grasp 学生需要掌握的概念
    

# =============================================================================
# Configuration
# 配置
# =============================================================================

# =============================================================================
# MedTutor-R1 Style Problem Decomposition Prompt
# MedTutor-R1风格的问题分解提示词
# Based on: https://github.com/Zhitao-He/MedTutor-R1
# =============================================================================

PROBLEM_DECOMPOSITION_PROMPT = """You are an expert clinical reasoning analyst. Your specialty is deconstructing complex medical problems into their core, logical, and learnable components.

Your task is to take a medical case, provided as a single JSON data point, and break down the entire diagnostic reasoning process into a series of essential, objective problem-solving steps. Your output must be completely neutral and analytical.

IMPORTANT: You must respond with valid JSON only.

## Key Generation Principles:

1. **Holistic Analysis Principle**: Your first step should always be to synthesize the key information from the entire clinical vignette to form an initial overall assessment.

2. **The Chain of Reasoning Principle**: The logical flow of your steps should generally follow the conceptual path of Observation → Interpretation → Conclusion. Think of this as a guiding framework for the flow of thought, not a rigid, fixed-step template.

3. **The Necessary Steps Principle**: Focus only on the most critical reasoning steps required to solve the problem. Avoid trivial, redundant, or irrelevant side-steps. Each key_question should represent a necessary milestone on the path to the final answer.

4. **The Complexity-Driven Step Count Principle**: 

   (1) The number of steps MUST be determined by the complexity of the problem. Do not force every problem into a fixed number of steps.

   (2) A simple identification task might only require 2 steps. A complex differential diagnosis with multiple findings might require 5 or more.

   (3) Your goal is to identify the most concise number of steps that are essential to logically and completely solve the problem.

## Output Format:
{
    "reasoning_steps": [
        {
            "step_id": "1",
            "key_question": "A specific, neutral question defining the sub-problem the student must think about",
            "step_summary": "A concise explanation of this step's purpose in the reasoning chain",
            "expected_understanding": "What the student should realize after thinking through this step"
        }
    ],
    "synthesis_step": "The final step that ties all reasoning together to reach the answer"
}

## Example for: "What advantage does endovascular repair have over open repair for ruptured AAA?"

{
    "reasoning_steps": [
        {
            "step_id": "1",
            "key_question": "What are the fundamental differences between endovascular repair (EVAR) and open surgical repair in terms of surgical approach and invasiveness?",
            "step_summary": "This establishes the baseline understanding of the two procedures being compared.",
            "expected_understanding": "EVAR is minimally invasive via catheter; open repair requires large abdominal incision."
        },
        {
            "step_id": "2",
            "key_question": "In the context of ruptured AAA, what are the immediate physiological consequences of each surgical approach on the patient?",
            "step_summary": "This step connects procedure differences to patient outcomes in the emergency setting.",
            "expected_understanding": "EVAR causes less blood loss, less trauma, shorter surgery time, less anesthesia stress."
        },
        {
            "step_id": "3",
            "key_question": "Based on clinical evidence (e.g., Kreienberg et al., J Vasc Surg 2013), how do 30-day mortality rates compare between EVAR and open repair for ruptured infrarenal AAA?",
            "step_summary": "This step brings in the specific outcome data that directly answers the question.",
            "expected_understanding": "EVAR has significantly lower 30-day mortality than open repair."
        },
        {
            "step_id": "4",
            "key_question": "What about long-term outcomes - how do 5-year survival rates compare between the two approaches?",
            "step_summary": "This addresses whether short-term benefits persist long-term.",
            "expected_understanding": "EVAR also shows better 5-year survival rates."
        }
    ],
    "synthesis_step": "Combining the understanding that EVAR is less invasive, causes less physiological stress, and has both lower 30-day mortality AND better 5-year survival, the advantage of EVAR over open repair becomes clear."
}

## CRITICAL RULES:
- Each key_question must be SPECIFIC to the medical content
- NO generic questions like "What is the main concept?"
- Steps should build logically toward the answer
- Do NOT reveal the answer directly in the questions
- Maintain complete neutrality: focus on analytical reasoning, not leading the student to a specific answer
- All questions must be objective and analytical, avoiding any subjective language or hints
"""

# Prompt for recursive decomposition when student struggles
# 当学生遇到困难时进行递归分解的提示词
RECURSIVE_DECOMPOSITION_PROMPT = """A student is struggling with this reasoning step. You must respond with valid JSON only.

Step: {step_question}
Expected Understanding: {expected_understanding}
Student's Response: {student_response}

The student doesn't seem to understand. Break this step down into SIMPLER sub-steps. The number of sub-steps should be determined by the complexity of the concept - use as many as needed to help the student understand.

## Rules:
1. Sub-steps must be EASIER than the original
2. Sub-steps should build up to the original understanding
3. Use specific medical terminology
4. Be encouraging
5. Generate as many sub-steps as necessary based on complexity (no hard limit)

## Response Format:
{
    "feedback": "Encouraging feedback acknowledging their effort",
    "missing_concept": "What specific concept they need to understand",
    "simpler_steps": [
        {
            "step_id": "{parent_id}.1",
            "key_question": "A simpler, more foundational question",
            "step_summary": "Why this simpler question helps",
            "expected_understanding": "What they should realize"
        },
        {
            "step_id": "{parent_id}.2",
            "key_question": "Another building-block question",
            "step_summary": "How this connects to the parent step",
            "expected_understanding": "What they should realize"
        }
        // Add more sub-steps as needed based on complexity
    ]
}
"""



# System prompt for evaluating answers (determines if decomposition needed)
# 评估答案的系统提示词（判断是否需要进一步分解）
EVALUATION_SYSTEM_PROMPT = """You are evaluating a student's answer to a medical sub-question. You must respond with valid JSON only.

## Your Task:
1. Check if the student's answer contains the key expected concepts
2. If correct: provide positive feedback
3. If incorrect: we will decompose further (handled separately)

## Response Format (JSON):
{
    "understood": true/false,
    "feedback": "Encouraging, specific feedback",
    "missing_concept": "What concept they need to grasp (if not understood)"
}

## Rules:
- "understood" means they demonstrate understanding of the key concepts
- Be specific about what they got right or wrong
- Be encouraging even when wrong
"""


# =============================================================================
# Helper Functions
# 辅助函数
# =============================================================================

def get_api_key() -> str:
    """
    Get OpenAI API key from environment variable or api-key.js file.
    从环境变量或api-key.js文件中获取OpenAI API密钥
    
    Returns:
        str: API key or empty string if not found
        返回：API密钥，如果未找到则返回空字符串
    """
    # Try environment variable first
    # 首先尝试从环境变量获取
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # Try api-key.js file
    # 如果环境变量中没有，尝试从api-key.js文件读取
    if not api_key:
        config_path = Path(__file__).parent / 'api-key.js'
        if config_path.exists():
            try:
                content = config_path.read_text(encoding='utf-8')
                match = re.search(r"OPENAI_API_KEY\s*[=:]\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    api_key = match.group(1)
            except Exception as e:
                print(f"Error reading api-key.js: {e}")
    
    return api_key or ""


def determine_hint_round(conversation_history: Optional[List[Dict]]) -> int:
    """
    Determine which round of hints based on conversation history.
    根据对话历史确定当前是第几轮提示
    
    Args:
        conversation_history: List of previous messages 之前的消息列表
        
    Returns:
        int: Round number (no hard limit) 返回轮次编号（无硬性限制）
    """
    if not conversation_history:
        return 1
    
    # Count how many hint rounds have been provided (check both old and new formats)
    # 统计已提供的提示轮次数量（检查新旧两种格式）
    hint_rounds = sum(1 for msg in conversation_history 
                      if msg.get("role") == "assistant" and 
                      ("knowledge_blocks" in str(msg.get("content", "")) or 
                       "hint_questions" in str(msg.get("content", ""))))
    
    # Return next round number (no hard limit)
    # 返回下一轮编号（无硬性限制）
    return hint_rounds + 1


def format_question_context(request: HintRequest) -> str:
    """
    Format the question and choices for the prompt.
    格式化问题和选项，用于构建提示词
    
    Args:
        request: HintRequest object 提示请求对象
        
    Returns:
        str: Formatted question context 格式化的问题上下文
    """
    context = f"""## Medical Question:
{request.question}

## Answer Choices:
"""
    for key in sorted(request.choices.keys()):
        context += f"{key}. {request.choices[key]}\n"
    
    context += f"""
## Student's Answer: {request.student_answer}
## Student's Choice: {request.choices.get(request.student_answer, 'Unknown')}
"""
    
    if request.source_context:
        context += f"""
## Relevant Medical Context (for your reference only - don't quote directly):
{request.source_context[:2000]}  # Limit context length 限制上下文长度
"""
    
    return context


# =============================================================================
# Main Generator Class
# 主要生成器类
# =============================================================================

class ProactiveQuestionGenerator:
    """
    Generates Socratic hints to guide students toward correct answers.
    生成苏格拉底式提示，引导学生找到正确答案
    
    Usage:
        用法示例：
        generator = ProactiveQuestionGenerator()
        
        request = HintRequest(
            question="What is the primary treatment for...",
            choices={"A": "...", "B": "...", "C": "...", "D": "..."},
            student_answer="B",
            correct_answer="A"
        )
        
        response = generator.generate_sub_questions(request)
        for node in response.tree.sub_nodes:
            print(f"[{node.id}] {node.question}")
            print(f"Tests: {node.knowledge_target}")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the generator with OpenAI client.
        使用OpenAI客户端初始化生成器
        
        Args:
            api_key: OpenAI API key (optional, will try to load from config)
            API密钥：OpenAI API密钥（可选，将尝试从配置加载）
        """
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or create api-key.js")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Use GPT-4o for best quality hints 使用GPT-4o以获得最佳质量的提示
    
    def generate_sub_questions(self, request: HintRequest) -> SocraticResponse:
        """
        Decompose a medical problem into reasoning steps (MedTutor-R1 style).
        将医学问题分解为推理步骤（MedTutor-R1风格）
        
        Based on clinical reasoning: Observation → Interpretation → Conclusion
        基于临床推理：观察 → 解释 → 结论
        
        Args:
            request: HintRequest containing question, choices, and student's answer
            请求对象：包含问题、选项和学生答案的HintRequest
            
        Returns:
            SocraticResponse containing the reasoning steps
            返回：包含推理步骤的SocraticResponse对象
        """
        # Build the prompt with full context
        # 构建包含完整上下文的提示词
        options_text = "\n".join([f"{k}: {v}" for k, v in request.choices.items()])
        
        user_prompt = f"""Decompose this medical question into reasoning steps:

## Question:
{request.question}

## Options:
{options_text}

## Correct Answer: {request.correct_answer}
(The student chose {request.student_answer}, which is wrong. Help them reason to the correct answer WITHOUT revealing it directly.)

{f"## Reference Context:\\n{request.source_context[:800]}" if request.source_context else ""}

Generate a chain of reasoning steps that will guide the student from observation to conclusion.
Each step should use SPECIFIC medical terms from this question.

Follow the format in my instructions exactly.
"""
        
        messages = [
            {"role": "system", "content": PROBLEM_DECOMPOSITION_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Parse reasoning steps
            # 解析推理步骤
            steps = []
            for step_data in result.get("reasoning_steps", []):
                steps.append(ReasoningStep(
                    step_id=str(step_data.get("step_id", len(steps) + 1)),
                    key_question=step_data.get("key_question", ""),
                    step_summary=step_data.get("step_summary", ""),
                    expected_understanding=step_data.get("expected_understanding", "")
                ))
            
            decomposition = ProblemDecomposition(
                original_question=request.question,
                reasoning_steps=steps,
                synthesis_step=result.get("synthesis_step", ""),
                current_step_index=0
            )
            
            return SocraticResponse(
                decomposition=decomposition,
                active_step_id=steps[0].step_id if steps else "1",
                total_steps=len(steps),
                is_complete=False
            )
            
        except Exception as e:
            # Fallback with basic reasoning steps
            # 异常时回退到基础推理步骤
            fallback_steps = [
                ReasoningStep("1", 
                    f"What are the key clinical features mentioned in this question about {request.question.split()[3] if len(request.question.split()) > 3 else 'this topic'}?",
                    "Identify the important clinical information before analyzing options.",
                    "The student should identify relevant clinical details."),
                ReasoningStep("2",
                    "How do these clinical features relate to the different answer options?",
                    "Connect observations to potential answers.",
                    "The student should see which options are supported by the evidence."),
                ReasoningStep("3",
                    "Based on this reasoning, which option is best supported?",
                    "Final synthesis of reasoning to reach a conclusion.",
                    "The student should identify the correct answer.")
            ]
            
            return SocraticResponse(
                decomposition=ProblemDecomposition(
                    original_question=request.question,
                    reasoning_steps=fallback_steps,
                    synthesis_step="Combine your observations and interpretations to identify the best answer.",
                    current_step_index=0
                ),
                active_step_id="1",
                total_steps=3,
                is_complete=False
            )
    
    def should_trigger_hints(
        self, 
        student_answer: str, 
        correct_answer: str, 
        user_message: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine if hints should be triggered.
        判断是否应该触发提示
        
        Args:
            student_answer: The answer selected by student 学生选择的答案
            correct_answer: The correct answer 正确答案
            user_message: Optional message from student (e.g., "give me a hint")
            可选的学生消息（例如："给我提示"）
            
        Returns:
            Tuple of (should_trigger: bool, reason: str)
            返回：元组（是否触发：布尔值，原因：字符串）
        """
        # Trigger 1: Wrong answer
        # 触发条件1：答案错误
        if student_answer != correct_answer:
            return True, "wrong_answer"
        
        # Trigger 2: Student asks for hints
        # 触发条件2：学生请求提示
        if user_message:
            hint_keywords = [
                "hint", "help", "clue", "guide", "stuck", 
                "don't understand", "confused", "explain",
                "提示", "帮助", "不懂", "不明白"  # Chinese keywords
            ]
            message_lower = user_message.lower()
            if any(keyword in message_lower for keyword in hint_keywords):
                return True, "student_request"
        
        return False, "no_trigger"
    
    def evaluate_response(
        self, 
        request: HintRequest,
        step: ReasoningStep,
        student_response: str
    ) -> EvaluationResult:
        """
        Evaluate the student's response to a reasoning step.
        If they don't show understanding, decompose into simpler sub-steps.
        评估学生对推理步骤的回答。
        如果学生没有表现出理解，则分解为更简单的子步骤。
        
        MedTutor-R1 approach: 观察 → 解释 → 结论
        MedTutor-R1方法：观察 → 解释 → 结论
        """
        # Evaluate understanding
        # 评估理解程度
        eval_prompt = f"""
## Reasoning Step:
Question: {step.key_question}
Purpose: {step.step_summary}
Expected Understanding: {step.expected_understanding}

## Student's Response:
{student_response}

Does the student's response show they understand the key concept?

Respond with JSON:
{{
    "understood": true/false,
    "feedback": "Encouraging, specific feedback",
    "missing_concept": "If not understood, what specific concept they need to grasp"
}}
"""
        
        messages = [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": eval_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            eval_result = json.loads(response.choices[0].message.content)
            understood = eval_result.get("understood", False)
            feedback = eval_result.get("feedback", "")
            missing = eval_result.get("missing_concept", "")
            
            if understood:
                # 如果理解了，返回成功结果
                return EvaluationResult(
                    step_id=step.step_id,
                    understood=True,
                    feedback=feedback,
                    sub_steps=[],
                    missing_concept=""
                )
            
            # Not understood - decompose into simpler sub-steps
            # 未理解 - 分解为更简单的子步骤
            decomp_prompt = RECURSIVE_DECOMPOSITION_PROMPT.format(
                step_question=step.key_question,
                expected_understanding=step.expected_understanding,
                student_response=student_response,
                parent_id=step.step_id
            )
            
            decomp_prompt += f"""

## Original Medical Context:
{request.question[:300]}

## What the student is missing:
{missing}

Generate simpler sub-steps to help them understand. The number of sub-steps should be determined by the complexity of the concept - use as many as needed (no hard limit).
"""
            
            decomp_messages = [
                {"role": "system", "content": "You are a Socratic medical tutor helping a struggling student."},
                {"role": "user", "content": decomp_prompt}
            ]
            
            decomp_response = self.client.chat.completions.create(
                model=self.model,
                messages=decomp_messages,
                temperature=0.7,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            decomp_result = json.loads(decomp_response.choices[0].message.content)
            
            # Parse simpler steps
            # 解析更简单的步骤
            sub_steps = []
            for s in decomp_result.get("simpler_steps", []):
                sub_steps.append(ReasoningStep(
                    step_id=s.get("step_id", f"{step.step_id}.{len(sub_steps)+1}"),
                    key_question=s.get("key_question", ""),
                    step_summary=s.get("step_summary", ""),
                    expected_understanding=s.get("expected_understanding", "")
                ))
            
            return EvaluationResult(
                step_id=step.step_id,
                understood=False,
                feedback=decomp_result.get("feedback", feedback),
                sub_steps=sub_steps,
                missing_concept=decomp_result.get("missing_concept", missing)
            )
            
        except Exception as e:
            # Fallback
            # 异常时回退处理
            return EvaluationResult(
                step_id=step.step_id,
                understood=False,
                feedback="Let me break this down into simpler steps.",
                sub_steps=[
                    ReasoningStep(f"{step.step_id}.1",
                        "What is the basic concept involved here?",
                        "Start with the fundamentals",
                        "Understand the basic definition"),
                    ReasoningStep(f"{step.step_id}.2",
                        "How does this concept apply to the clinical scenario?",
                        "Connect the concept to the specific case",
                        "See how the concept explains the situation")
                ],
                missing_concept=""
            )
    
    def format_hints_for_display(self, response: SocraticResponse) -> str:
        """
        Format the reasoning steps for display in the UI (MedTutor-R1 style).
        格式化推理步骤以便在UI中显示（MedTutor-R1风格）
        
        Args:
            response: SocraticResponse object SocraticResponse对象
            
        Returns:
            str: Formatted text for display 格式化后的显示文本
        """
        decomp = response.decomposition
        
        output = f"""## 🧠 Clinical Reasoning Chain

**Question:** {decomp.original_question[:100]}...

---

**Work through each reasoning step:**

"""
        
        for i, step in enumerate(decomp.reasoning_steps):
            # Determine status icon
            # 确定状态图标
            status = ""
            if step.is_understood is True:
                status = " ✅"
            elif step.is_understood is False:
                status = " 🔄"  # Needs more work 需要更多努力
            else:
                status = " ⏳"  # Not yet answered 尚未回答
            
            active = " ← **Current Step**" if step.step_id == response.active_step_id else ""
            
            output += f"""### Step {step.step_id}: {step.key_question}{status}{active}
*{step.step_summary}*

"""
            # If there are sub-steps (from decomposition), show them
            # 如果有子步骤（来自分解），显示它们
            if step.sub_steps:
                output += "**Breaking this down:**\n\n"
                for sub in step.sub_steps:
                    sub_status = "✅" if sub.is_understood else "⏳"
                    output += f"""   - **[{sub.step_id}]** {sub.key_question} {sub_status}
     *{sub.step_summary}*

"""
        
        output += f"""---
**🎯 Final Synthesis:** {decomp.synthesis_step}
"""
        
        return output
    
    def step_to_dict(self, step: ReasoningStep) -> Dict:
        """Convert ReasoningStep to serializable dict.
        将ReasoningStep转换为可序列化的字典"""
        return {
            "step_id": step.step_id,
            "key_question": step.key_question,
            "step_summary": step.step_summary,
            "expected_understanding": step.expected_understanding,
            "student_response": step.student_response,
            "is_understood": step.is_understood,
            "feedback": step.feedback,
            "sub_steps": [self.step_to_dict(s) for s in step.sub_steps]
        }
    
    def decomposition_to_dict(self, decomp: ProblemDecomposition) -> Dict:
        """Convert ProblemDecomposition to serializable dict.
        将ProblemDecomposition转换为可序列化的字典"""
        return {
            "original_question": decomp.original_question,
            "reasoning_steps": [self.step_to_dict(s) for s in decomp.reasoning_steps],
            "synthesis_step": decomp.synthesis_step,
            "current_step_index": decomp.current_step_index
        }
    
    def to_dict(self, response: SocraticResponse) -> Dict:
        """
        Convert SocraticResponse to dictionary for JSON serialization.
        将SocraticResponse转换为字典以便JSON序列化
        
        Args:
            response: SocraticResponse object SocraticResponse对象
            
        Returns:
            dict: Dictionary representation 字典表示
        """
        return {
            "decomposition": self.decomposition_to_dict(response.decomposition),
            "active_step_id": response.active_step_id,
            "total_steps": response.total_steps,
            "is_complete": response.is_complete
        }


# =============================================================================
# Convenience Functions
# 便捷函数
# =============================================================================

def generate_socratic_questions(
    question: str,
    choices: Dict[str, str],
    student_answer: str,
    correct_answer: str,
    explanations: Optional[Dict[str, str]] = None,
    conversation_history: Optional[List[Dict]] = None,
    source_context: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    Convenience function to generate Socratic reasoning steps without instantiating the class.
    便捷函数：无需实例化类即可生成苏格拉底式推理步骤
    
    Args:
        question: The medical question 医学问题
        choices: Dictionary of answer options 答案选项字典
        student_answer: Student's selected answer 学生选择的答案
        correct_answer: The correct answer 正确答案
        explanations: Optional explanations for each option 每个选项的可选解释
        conversation_history: Previous hint rounds 之前的提示轮次
        source_context: RAG retrieved context RAG检索到的上下文
        api_key: Optional API key 可选的API密钥
        
    Returns:
        dict: Problem decomposition as dictionary 问题分解的字典表示
    """
    generator = ProactiveQuestionGenerator(api_key=api_key)
    
    request = HintRequest(
        question=question,
        choices=choices,
        student_answer=student_answer,
        correct_answer=correct_answer,
        explanations=explanations,
        conversation_history=conversation_history,
        source_context=source_context
    )
    
    response = generator.generate_sub_questions(request)
    return generator.to_dict(response)


def check_hint_trigger(
    student_answer: str,
    correct_answer: str,
    user_message: Optional[str] = None
) -> Dict:
    """
    Check if hints should be triggered.
    检查是否应该触发提示
    
    Args:
        student_answer: Student's selected answer 学生选择的答案
        correct_answer: The correct answer 正确答案
        user_message: Optional user message 可选的用户消息
        
    Returns:
        dict: {"should_trigger": bool, "reason": str}
        返回：字典 {"should_trigger": 布尔值, "reason": 字符串}
    """
    generator = ProactiveQuestionGenerator()
    should_trigger, reason = generator.should_trigger_hints(
        student_answer, correct_answer, user_message
    )
    return {"should_trigger": should_trigger, "reason": reason}


# =============================================================================
# Main Entry Point (for testing)
# 主入口点（用于测试）
# =============================================================================

if __name__ == "__main__":
    # Example usage - MedTutor-R1 style problem decomposition
    # 示例用法 - MedTutor-R1风格的问题分解
    print("=" * 60)
    print("MedTutor-R1 Style Problem Decomposition - Test")
    print("=" * 60)
    
    # Example medical question
    # 示例医学问题
    test_request = HintRequest(
        question="What advantage does endovascular repair of ruptured infrarenal abdominal aortic aneurysms have over open surgical repair?",
        choices={
            "A": "Lower 30-day mortality and better 5-year survival rates",
            "B": "Higher complication rates",
            "C": "Increased need for blood transfusions",
            "D": "Longer recovery times"
        },
        student_answer="B",  # Wrong answer 错误答案
        correct_answer="A",
        source_context="Kreienberg PB, et al. Endovascular repair of ruptured infrarenal abdominal aortic aneurysm is associated with lower 30-day mortality and better 5-year survival rates than open surgical repair. J Vasc Surg 2013;57:368-75."
    )
    
    try:
        generator = ProactiveQuestionGenerator()
        
        # Generate reasoning steps
        # 生成推理步骤
        print("\n--- Clinical Reasoning Chain ---")
        response = generator.generate_sub_questions(test_request)
        
        decomp = response.decomposition
        print(f"Original Question: {decomp.original_question[:80]}...")
        print(f"Total Steps: {response.total_steps}")
        print()
        
        for step in decomp.reasoning_steps:
            print(f"[Step {step.step_id}] {step.key_question}")
            print(f"    Purpose: {step.step_summary}")
            print(f"    Expected: {step.expected_understanding}")
            print()
        
        print(f"Synthesis: {decomp.synthesis_step}")
        
        # Show formatted display
        # 显示格式化输出
        print("\n--- Formatted Display ---")
        print(generator.format_hints_for_display(response))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure your API key is configured in api-key.js or OPENAI_API_KEY environment variable")
        print("请确保在api-key.js文件或OPENAI_API_KEY环境变量中配置了API密钥")

