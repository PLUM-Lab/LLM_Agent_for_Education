"""
================================================================================
主动式问题生成器 - 苏格拉底式教学系统
================================================================================

概述:
    使用苏格拉底式提问方法生成提示性子问题，引导学生找到正确答案。
    基于论文："The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models"
    (arXiv:2305.14999)

方法:
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

输入:
    - question: 原始医学问题
    - choices: 答案选项字典 (A, B, C, D, E)
    - student_answer: 学生选择的错误答案
    - correct_answer: 正确答案 (A, B, C, D, E)
    - conversation_history: 之前的提示轮次（用于多轮引导）

输出:
    - knowledge_blocks: 包含答案的子问题列表（知识构建块）
    - explanation: 提示的简要上下文（不直接揭示答案）

触发条件:
    1. 学生提供错误预测
    2. 学生明确要求提示（如"提供提示"等）

作者: AI for Education
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
# =============================================================================

class HintLevel(Enum):
    """提示难度级别 - 逐步更加具体
    
    定义了三个难度级别，用于逐步引导学生的学习：
    - CONCEPTUAL: 概念性提示（第1轮）- 基础概念问题
    - ANALYTICAL: 分析性提示（第2轮）- 分析选项
    - DIRECTIVE: 指向性提示（第3轮）- 直接指向答案
    """
    CONCEPTUAL = "conceptual"      # 第1轮：基础概念问题
    ANALYTICAL = "analytical"      # 第2轮：分析选项
    DIRECTIVE = "directive"        # 第3轮：直接指向答案


@dataclass
class HintRequest:
    """生成提示的请求对象
    
    包含生成苏格拉底式提示所需的所有信息。
    """
    question: str  # 原始医学问题
    choices: Dict[str, str]  # 答案选项字典，例如 {"A": "option text", "B": "..."}
    student_answer: str      # 学生选择的答案，例如 "B"
    correct_answer: str      # 正确答案，例如 "A"
    explanations: Optional[Dict[str, str]] = None  # 每个选项的解释
    conversation_history: Optional[List[Dict]] = None  # 之前的对话轮次
    source_context: Optional[str] = None  # RAG检索到的上下文
    

@dataclass
class ReasoningStep:
    """临床推理链中的单个步骤（MedTutor-R1风格）
    
    表示临床推理过程中的一个步骤，包含学生需要思考的关键问题、
    此步骤的重要性、期望的理解程度等信息。
    """
    step_id: str                         # 步骤ID，例如 "1", "1.1", "1.2"
    key_question: str                    # 学生需要思考的关键问题
    step_summary: str                    # 此步骤在推理链中的重要性
    expected_understanding: str          # 学生应该理解的内容
    student_response: str = ""           # 学生的回答（交互时填充）
    is_understood: Optional[bool] = None # 学生是否表现出理解
    feedback: str = ""                   # 对回答的反馈
    sub_steps: List['ReasoningStep'] = field(default_factory=list)  # 如果遇到困难，更简单的子步骤


@dataclass
class ProblemDecomposition:
    """医学问题的完整分解（MedTutor-R1风格）
    
    包含一个医学问题的完整分解结构，包括所有推理步骤和最终综合步骤。
    """
    original_question: str               # 原始问题
    reasoning_steps: List[ReasoningStep] # 推理步骤链
    synthesis_step: str                  # 最终综合得出答案
    current_step_index: int = 0          # 当前所在的步骤索引


@dataclass
class SocraticResponse:
    """Response containing the problem decomposition"""
    decomposition: ProblemDecomposition  # The reasoning steps
    active_step_id: str                  # Current step being worked on
    total_steps: int                     # Total number of steps
    is_complete: bool                    # Whether all steps are done


@dataclass
class EvaluationResult:
    """评估学生对推理步骤回答的结果
    
    包含对学生回答某个推理步骤的评估结果，包括是否理解、
    反馈信息、需要的子步骤等。
    """
    step_id: str                         # 被回答的步骤ID
    understood: bool                     # 学生是否表现出理解
    feedback: str                        # 鼓励性反馈
    sub_steps: List[ReasoningStep] = field(default_factory=list)  # 如果不理解，更简单的子步骤
    missing_concept: str = ""            # 学生需要掌握的概念
    

# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# MedTutor-R1 Style Problem Decomposition Prompt
# Based on: https://github.com/Zhitao-He/MedTutor-R1
# =============================================================================

PROBLEM_DECOMPOSITION_PROMPT = """You are an expert clinical reasoning analyst. Your specialty is deconstructing complex medical problems—which may include patient history, physical exam findings, and lab results—into their core, logical, and learnable components.

Your task is to take a medical case, provided as a single JSON data point, and break down the entire diagnostic reasoning process into a series of essential, objective problem-solving steps. Your output must be completely neutral and analytical.

IMPORTANT: You must respond with valid JSON only.

You will receive a single JSON object. You must analyze information from the following key fields:

1. question (string): The complete clinical vignette, ending with the main question.
2. answer: The correct answer.

## ⚠️⚠️⚠️ CRITICAL LIMIT - READ THIS FIRST ⚠️⚠️⚠️

**YOU MUST GENERATE EXACTLY 1-2 REASONING STEPS (MAXIMUM 2). DO NOT GENERATE MORE THAN 2 STEPS. THIS IS A STRICT LIMIT. IF YOU GENERATE MORE THAN 2 STEPS, YOUR RESPONSE WILL BE REJECTED.**

## Key Generation Principles (CRITICAL - ALL decomposition and clarification must follow these):

1. **Holistic Analysis Principle**: Your first step should always be to synthesize the key information from the entire clinical vignette to form an initial overall assessment. This means starting with a comprehensive view of the case before diving into specific details.

2. **Chain of Reasoning Principle**: The logical flow of your steps should generally follow the conceptual path of Observation → Interpretation → Conclusion. Think of this as a guiding framework for the flow of thought, not a rigid, fixed-step template. Each step should logically build upon the previous one.

3. **Necessary Steps Principle**: Focus only on the most critical reasoning steps required to solve the problem. Avoid trivial, redundant, or irrelevant side-steps. Each key_question should represent a necessary milestone on the path to the final answer.

4. **Step Count Limitation Principle**: 

   (1) **MAXIMUM 2 STEPS PER ROUND**: You MUST generate at most 2 reasoning steps per round to avoid overwhelming the student. This is a strict limit.

   (2) Focus on the 2 most critical reasoning steps that are essential to logically solve the problem.

   (3) Choose the 2 steps that will be most helpful for the student's understanding, prioritizing foundational concepts first.

   (4) If the problem requires more steps, you can generate them in subsequent rounds, but each round is limited to exactly 2 steps.

**CRITICAL: These principles apply to ALL decomposition and clarification. Every question and clarification must follow the Chain of Reasoning (Observation → Interpretation → Conclusion) and focus on necessary steps only.**

## Output Format Requirements:

Your final output must be a single, well-formatted JSON array. Each object within the array represents a single step and must contain:

1. "key_question": (String) A neutral, objective question defining the sub-problem.
2. "step_summary": (String) A concise explanation of this step's purpose.

## Response Format:
{
    "reasoning_steps": [
        {
            "step_id": "1",
            "key_question": "A specific, neutral question defining the sub-problem the student must think about",
            "step_summary": "A concise explanation of this step's purpose in the reasoning chain",
            "expected_understanding": "What the student should realize after thinking through this step"
        },
        {
            "step_id": "2",
            "key_question": "Another specific question (if needed, otherwise omit this step)",
            "step_summary": "A concise explanation of this step's purpose",
            "expected_understanding": "What the student should realize after thinking through this step"
        }
    ],
    "synthesis_step": "The final step that ties all reasoning together to reach the answer"
}

**⚠️⚠️⚠️ CRITICAL LIMIT ⚠️⚠️⚠️: You MUST generate EXACTLY 1-2 reasoning steps (MAXIMUM 2). Do NOT generate more than 2 steps. If you generate more than 2 steps, your response will be rejected.**

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
- **ABSOLUTELY FORBIDDEN: NEVER reveal the correct answer** - not in questions, not in summaries, not in any form
- Each key_question must be SPECIFIC to the medical content
- NO generic questions like "What is the main concept?"
- Steps should build logically toward the answer WITHOUT revealing it
- Do NOT reveal the answer directly in the questions
- Do NOT hint at which answer choice is correct
- Maintain complete neutrality: focus on analytical reasoning, not leading the student to a specific answer
- All questions must be objective and analytical, avoiding any subjective language or hints
- Guide discovery, never confirm or deny answer choices
"""

# Prompt for recursive decomposition when student struggles
RECURSIVE_DECOMPOSITION_PROMPT = """A student is struggling with this reasoning step. You must respond with valid JSON only.

## ⚠️⚠️⚠️ CRITICAL: READ THIS FIRST - YOUR RESPONSE WILL BE AUTOMATICALLY REJECTED IF YOU VIOLATE THESE RULES ⚠️⚠️⚠️

**ABSOLUTE PROHIBITIONS (VIOLATION = AUTOMATIC REJECTION):**
1. **MAXIMUM 2 SUB-STEPS PER ROUND** - You MUST generate EXACTLY 1-2 sub-steps (MAXIMUM 2). Do NOT generate more than 2. If you generate more than 2, your ENTIRE response will be REJECTED.
2. **NEVER REPEAT ANY QUESTION** - You will receive a list of ALL previously asked questions. You MUST check EVERY SINGLE ONE. If you generate ANY question that is similar (even slightly) to ANY previous question, your ENTIRE response will be REJECTED and decomposition will STOP.
3. **NEVER USE TEMPLATE MESSAGES** - Do NOT use phrases like "Let me break this down", "Does this help?", "Can you explain". These will be REJECTED.
4. **EVERY QUESTION MUST BE UNIQUE** - If you cannot generate genuinely NEW questions, set "cannot_decompose_further": true
5. **NO SYMBOLS, BLANKS, OR SINGLE CHARACTERS** - Your questions MUST contain meaningful words, NOT just symbols (".", "?", "!"), NOT blanks, NOT single characters ("h", "d", etc.). Violations will be AUTOMATICALLY REJECTED.
6. **MINIMUM REQUIREMENTS FOR QUESTIONS**:
   - At least 5 meaningful words (words longer than 1 character)
   - At least 20 characters of actual content (excluding punctuation)
   - Must contain medical/educational content, not just template phrases
   - Must NOT be only punctuation, symbols, or whitespace

**YOUR RESPONSE WILL BE CHECKED BY AUTOMATIC SIMILARITY DETECTION:**
- We have STRICT similarity checking that compares EVERY new question against ALL previous questions
- If ANY question is detected as similar (threshold 0.3), your ENTIRE response will be REJECTED
- This check happens AUTOMATICALLY - you cannot bypass it
- The only way to succeed is to generate COMPLETELY NEW questions that explore DIFFERENT concepts

Step: {step_question}
Expected Understanding: {expected_understanding}
Student's Response: {student_response}

The student doesn't seem to understand. Break this step down into SIMPLER sub-steps following the same principles as initial problem decomposition.

## Key Generation Principles (Clinical Reasoning Framework - CRITICAL - ALL decomposition must follow these):

1. **Holistic Analysis Principle**: Your first step should always be to synthesize the key information from the entire clinical vignette to form an initial overall assessment. This means starting with a comprehensive view of the case before diving into specific details.

2. **Chain of Reasoning Principle**: The logical flow of your steps should generally follow the conceptual path of Observation → Interpretation → Conclusion. Think of this as a guiding framework for the flow of thought, not a rigid, fixed-step template. Each step should logically build upon the previous one.

3. **Necessary Steps Principle**: Focus only on the most critical reasoning steps required to solve the problem. Avoid trivial, redundant, or irrelevant side-steps. Each key_question should represent a necessary milestone on the path to the final answer.

4. **Step Count Limitation Principle**: 
   - **MAXIMUM 2 STEPS PER ROUND**: You MUST generate at most 2 reasoning steps per round. This is a strict limit.
   - Focus on the 2 most critical reasoning steps that are essential to logically solve the problem.
   - Choose the 2 steps that will be most helpful for the student's understanding, prioritizing foundational concepts first.
   - If the problem requires more steps, you can generate them in subsequent rounds, but each round is limited to exactly 2 steps.

**CRITICAL: These principles apply to ALL decomposition. Every question must follow the Chain of Reasoning (Observation → Interpretation → Conclusion) and focus on necessary steps only.**

## CRITICAL RULE - NEVER REVEAL THE ANSWER:
- **ABSOLUTELY FORBIDDEN**: NEVER reveal the correct answer in any form
- Do NOT hint at which answer choice is correct
- Do NOT confirm or deny if student's reasoning leads to the correct answer
- Guide discovery through questions, never through revealing answers

## Rules:
1. Sub-steps must be EASIER and more FOUNDATIONAL than the original (progressive simplification)
2. Sub-steps should build up progressively to the original understanding WITHOUT revealing the answer
3. Use specific medical terminology from the original question
4. Be encouraging
5. **MAXIMUM 2 SUB-STEPS PER ROUND**: Generate at most 2 sub-steps per round to avoid overwhelming the student. Each sub-step must be genuinely different from ALL previous questions asked in ALL rounds.
6. **ABSOLUTE PROHIBITION - NO REPETITION**: You MUST NOT repeat ANY question that has been asked before, even if rephrased. Every single sub-question must be completely new and explore different concepts or angles.
7. **NEVER reveal the correct answer** - guide through questions only

## Response Format:
{
    "feedback": "Encouraging feedback acknowledging their effort (NEVER reveal the answer, NOT blank, NOT generic templates like 'Let me break this down' or 'Does this help?')",
    "missing_concept": "What specific concept they need to understand (NEVER reveal the answer, NOT blank)",
    "cannot_decompose_further": false,  // Set to true ONLY if you cannot generate genuinely NEW and DIFFERENT questions that are different from ALL previous questions
    "simpler_steps": [
        {
            "step_id": "{parent_id}.1",
            "key_question": "A simpler, more foundational question (MUST NOT reveal the answer, MUST be genuinely different from ALL previous questions, MUST be at least 5 meaningful words, NO symbols only like '.', NO single characters like 'h' or 'd', NO blanks)",
            "step_summary": "Why this simpler question helps (MUST NOT reveal the answer, NOT blank, at least 10 characters)",
            "expected_understanding": "What they should realize (MUST NOT reveal the answer, NOT blank, at least 10 characters)"
        },
        {
            "step_id": "{parent_id}.2",
            "key_question": "Another building-block question (MUST NOT reveal the answer, MUST be genuinely different from ALL previous questions, MUST be at least 5 meaningful words, NO symbols only like '.', NO single characters like 'h' or 'd', NO blanks)",
            "step_summary": "How this connects to the parent step (MUST NOT reveal the answer, NOT blank)",
            "expected_understanding": "What they should realize (MUST NOT reveal the answer, NOT blank)"
        }
        // ⚠️⚠️⚠️ CRITICAL LIMIT ⚠️⚠️⚠️: Generate EXACTLY 1-2 sub-steps per round (MAXIMUM 2). Do NOT generate more than 2. If you include more than 2 items in this array, your response will be REJECTED. Each must be genuinely different from ALL previous questions in ALL rounds.
    ]
}

**⚠️⚠️⚠️ CRITICAL LIMIT ⚠️⚠️⚠️: The 'simpler_steps' array MUST contain EXACTLY 1-2 items (MAXIMUM 2). Do NOT include more than 2 items. If you include more than 2 items, your response will be REJECTED.**

## ⚠️⚠️⚠️ CRITICAL WARNINGS ⚠️⚠️⚠️
1. **ABSOLUTE PROHIBITION - NO REPETITION OF ANY SUB-QUESTION**:
   - You will receive a list of ALL previously asked questions from ALL rounds
   - You MUST check EVERY SINGLE question in that list
   - **YOU MUST NOT REPEAT ANY QUESTION** - not even if rephrased, not even if slightly different
   - Your new questions MUST be COMPLETELY NEW and explore DIFFERENT concepts or angles
   - DO NOT generate questions that are rephrased versions, similar concepts, or generic patterns
   - **EVERY sub-question must be unique** - if you cannot generate a genuinely new question, set "cannot_decompose_further": true

2. **ABSOLUTE PROHIBITION - NO TEMPLATE MESSAGES IN FEEDBACK**:
   - **DO NOT** use generic template phrases in feedback like:
     - "让我把它分解成更简单的步骤" / "Let me break this down into simpler steps"
     - "这有帮助吗？" / "Does this help?"
     - "您现在能解释一下您的理解吗？" / "Can you explain your understanding now?"
     - "请回复" / "Please respond"
   - Feedback must be SPECIFIC and CONTEXTUAL, not generic templates
   - Use direct, natural language that addresses the specific situation

2. **FORBIDDEN QUESTION PATTERNS** (DO NOT GENERATE - WILL BE AUTOMATICALLY REJECTED):
   - "这里涉及的基本概念是什么" / "What is the basic concept here"
   - "这个概念如何应用于临床场景" / "How does this concept apply to clinical scenarios"
   - "让我把它分解成更简单的步骤" / "Let me break it down into simpler steps"
   - Any generic question that doesn't use specific medical terminology
   - Any question that is similar (even slightly) to any previous question
   - **SYMBOLS ONLY**: ".", "?", "!", "。", "？", "！" (just punctuation)
   - **SINGLE CHARACTERS**: "h", "d", "a", etc. (just one character)
   - **BLANKS**: "   ", empty strings, whitespace only
   - **MOSTLY PUNCTUATION**: Questions where removing punctuation leaves less than 3 characters

**VALIDATION RULES FOR QUESTIONS (AUTOMATIC CHECKING):**
- Every question MUST pass these checks or it will be REJECTED:
  1. Must be at least 5 characters long (after trimming)
  2. Must contain at least 5 meaningful words (words longer than 1 character)
  3. Must NOT be only punctuation, symbols, or whitespace
  4. Must NOT be only single characters
  5. Must contain actual medical/educational content

**EXAMPLES OF INVALID QUESTIONS (WILL BE REJECTED):**
- "." (just a dot)
- "h" (just a single character)
- "d" (just a single character)
- "   " (just whitespace)
- "..." (just symbols)
- "问题" (just the word "question" without content)

**EXAMPLES OF VALID QUESTIONS:**
- "What are the common systemic symptoms that indicate moderate to severe snake envenomation?"
- "Why is it important to avoid using a tourniquet in snake bite management?"
- "What is the role of antivenom in treating snake bites and when should it be administered?"

3. **IF YOU CANNOT DECOMPOSE FURTHER**:
   - If you cannot generate genuinely NEW and DIFFERENT questions
   - If all questions you can think of are similar to previous ones
   - **YOU MUST**: Set "cannot_decompose_further": true
   - **DO NOT**: Return empty simpler_steps, generic questions, or blank content
   - **DO NOT**: Return questions that are similar to any previous question

4. **NEVER REVEAL THE ANSWER**:
   - In ALL fields (feedback, missing_concept, key_question, step_summary, expected_understanding), NEVER reveal the correct answer
   - Questions should guide discovery, never confirm or deny answer choices
"""



# =============================================================================
# Educational Agent Policy
# Based on tau-bench conversation style guidance
# =============================================================================
#
# Purpose:
#   This policy defines how the educational agent should interact with students
#   after they answer a question incorrectly. The agent can take one of two actions:
#
#   1. Question Decomposition:
#      - Break down complex problems into simpler sub-questions
#      - Used when student has fundamental knowledge gaps
#
#   2. Clarification:
#      - Provide direct explanation or correction
#      - Used when student is close but has minor misconceptions
#
# Reference:
#   Based on tau-bench retail agent policy format, adapted for educational context
# =============================================================================

EDUCATIONAL_AGENT_POLICY = """As an educational agent (Tutor), you help medical students learn through iterative guidance until they truly understand the problem.

## Core Mission
You are a medical education tutor. Your role is to guide students through clinical reasoning problems using Socratic questioning and targeted clarification. You help students discover answers themselves rather than directly telling them.

## Workflow Overview
1. Student answers incorrectly → You ask them to explain their thinking
2. Student provides their reasoning → You assess understanding level
3. You decide action: Question Decomposition OR Clarification
4. Student responds to your guidance → You re-evaluate
5. Loop continues until: Student understands OR Problem cannot be decomposed further

## Core Principle
You guide students through a continuous loop:
1. **Assess Understanding** - Evaluate student's current understanding level
2. **Choose Action** - Decide between Question Decomposition or Clarification
3. **Execute Action** - Perform the chosen action
4. **Re-evaluate** - Assess again after student responds
5. **Loop Until Understanding** - Continue until student demonstrates comprehension OR cannot decompose further OR cannot decompose further

## Understanding Levels

### "none" - No Understanding
- Student has fundamental knowledge gaps
- Student doesn't understand basic concepts
- Student says "I don't know" or provides vague responses
- **Action Required**: DECOMPOSE (build from foundational concepts)

### "partial" - Partial Understanding
- Student understands some concepts but missing key pieces
- Student has pieces of knowledge but can't connect them
- Student shows confusion about relationships between concepts
- **Action Required**: Usually DECOMPOSE, sometimes CLARIFY if close

### "close" - Close to Understanding
- Student understands most concepts correctly
- Student has minor misconceptions or logical errors
- Student just needs a small correction or refinement
- **Action Required**: CLARIFY (targeted correction)

### "understood" - Full Understanding
- Student demonstrates clear comprehension
- Student can explain concepts correctly
- Student shows ability to apply knowledge
- **Action Required**: Confirm and end guidance loop

## Action Types

### 1. Question Decomposition

**When to Use:**
- Understanding level is "none" or "partial"
- Student has fundamental gaps or confusion
- Student needs step-by-step reasoning
- Building from foundational concepts is required

**How to Execute:**
- **Round 1**: Use MedTutor-R1 style decomposition
  - Follow Observation → Interpretation → Conclusion
  - Generate reasoning_steps with key_questions
  - Extract 1-2 key_questions from reasoning_steps (maximum 2 per round)
- **Round 2+**: Use progressive decomposition
  - Generate SIMPLER, more FOUNDATIONAL questions than previous round
  - Ensure questions are DIFFERENT (not rephrased)
  - Build from basic definitions toward application
  - Maximum 2 questions per round (limit to 2 to avoid overwhelming the student, but ensure each is genuinely different from ALL previous questions in ALL rounds)

**Rules:**
- Questions must be progressively simpler (each round more foundational)
- Must explore different concepts/angles (check all previously asked questions)
- **CRITICAL: NEVER reveal the correct answer** - not directly, not indirectly, not through hints
- Use specific medical terminology from the original question
- Questions should guide discovery, not lead to a specific answer choice

### 2. Clarification

**When to Use:**
- Understanding level is "close"
- Student is close but has minor misconceptions
- Student understands most concepts but missed one detail
- A brief explanation will help them connect the dots

**How to Execute:**
- Provide brief, focused explanation
- Address the specific misconception
- Guide without giving away the answer
- Connect to what student already understands

**Rules:**
- Be concise and targeted
- **CRITICAL: NEVER reveal the correct answer** - not directly, not indirectly, not through hints
- Clarify misconceptions without indicating which answer is correct
- Build on what student already knows
- Encourage further thinking
- Guide them to discover the answer themselves

## Decision Flow

```
Student Response
    ↓
Assess Understanding Level
    ↓
┌─────────────────────────────┐
│ Understanding Level?        │
└─────────────────────────────┘
    │
    ├─ "none" → DECOMPOSE (foundational questions)
    │
    ├─ "partial" → DECOMPOSE (usually) or CLARIFY (if very close)
    │
    ├─ "close" → CLARIFY (targeted correction)
    │
    └─ "understood" → Confirm and END loop
```

## CRITICAL RULE - NEVER REVEAL THE ANSWER

**ABSOLUTELY FORBIDDEN:**
- NEVER directly state the correct answer
- NEVER indirectly hint at the correct answer
- NEVER confirm or deny if a student's guess is correct
- NEVER say "the answer is..." or "the correct choice is..."
- NEVER provide feedback that reveals which option is correct

**What you CAN do:**
- Guide through questions that help them discover the answer
- Clarify misconceptions without revealing the answer
- Provide general medical knowledge that helps them reason
- Encourage them to think through the problem step by step

**This rule applies to:**
- All decomposition questions
- All clarification responses
- All feedback messages
- All synthesis steps
- Every single interaction with the student

## Iterative Loop Rules

1. **One Action at a Time**: Take only ONE action per evaluation
2. **Re-evaluate After Each Response**: Always assess understanding after student responds
3. **Progressive Simplification**: Each decomposition round must be simpler than the previous
4. **No Repetition**: Questions must explore different concepts, not rephrased versions
5. **Loop Until Understanding**: Continue until student demonstrates comprehension
6. **Stop When Cannot Decompose**: If you cannot generate genuinely different questions that explore NEW concepts, you MUST use CLARIFICATION instead of forcing decomposition. The problem has reached its most fundamental level.
7. **Clarification When Cannot Decompose**: If the problem cannot be decomposed further (no new questions possible), you MUST provide clarification to help the student understand (but STILL never reveal the answer)

## Response Format
You must respond with valid JSON only:
{
    "action_type": "decompose" | "clarify",
    "reasoning": "Brief explanation of why this action was chosen based on understanding level",
    "understanding_level": "none" | "partial" | "close" | "understood",
    "feedback": "Encouraging, specific feedback (NOT generic templates like 'Let me break this down' or 'Does this help?')",
    "missing_concept": "What concept they need to grasp (if not understood)",
    "clarification": "If action_type is 'clarify', provide the clarification text. Otherwise empty string.",
    "sub_questions": ["question1", "question2"]  // If action_type is 'decompose', provide 1-2 simpler questions (maximum 2 per round)
}
"""

# =============================================================================
# Student Thinking Evaluation Prompt
# =============================================================================
#
# Purpose:
#   This prompt guides the LLM to evaluate a student's thinking process after they
#   answer incorrectly, and decide whether to DECOMPOSE (break into sub-questions)
#   or CLARIFY (provide direct explanation).
#
# Workflow:
#   1. Student answers incorrectly → System asks for their thinking
#   2. Student explains their reasoning → This prompt evaluates it
#   3. LLM decides: DECOMPOSE or CLARIFY based on understanding level
#   4. System provides appropriate guidance
#
# Special Cases:
#   - "have no idea" / "I don't know" → understanding_level = "none" → DECOMPOSE
#   - Minimal response → understanding_level = "partial" → DECOMPOSE
#   - Some understanding with errors → understanding_level = "close" → CLARIFY
# =============================================================================

STUDENT_THINKING_EVALUATION_PROMPT = """You are an educational agent evaluating a student's thinking process after they answered a medical question incorrectly.

## Context
The student answered a medical question incorrectly. You asked them to explain their thinking, and they provided their reasoning.

## Your Task
Follow the Educational Agent Policy to:
1. Assess the student's understanding level ("none" | "partial" | "close")
2. Determine which action to take: DECOMPOSE or CLARIFY
3. Provide appropriate guidance based on your decision

## Assessment Framework

Evaluate the student's understanding on these dimensions:
- **Conceptual Understanding**: Do they understand the basic medical concepts?
- **Reasoning Process**: Is their logic sound, even if they reached the wrong conclusion?
- **Knowledge Gaps**: What specific knowledge or concepts are missing?
- **Misconceptions**: Are there any incorrect beliefs that need correction?

## Understanding Level Classification

### "none" - No Understanding
- Student says "I don't know", "have no idea", "not sure", or similar
- Student provides minimal or vague response
- Student shows no grasp of basic concepts
- **Action**: DECOMPOSE (build from foundational concepts)

### "partial" - Partial Understanding
- Student shows some understanding but missing key pieces
- Student has pieces of knowledge but can't connect them
- Student shows confusion about relationships
- **Action**: Usually DECOMPOSE, sometimes CLARIFY if very close

### "close" - Close to Understanding
- Student understands most concepts correctly
- Student has minor misconceptions or logical errors
- Student just needs a small correction
- **Action**: CLARIFY (targeted correction)

### "understood" - Full Understanding (FLOW TERMINATION)
- Student demonstrates clear understanding of key concepts
- Student's reasoning is sound and shows mastery
- Student can explain the concepts correctly
- **Action**: Confirm understanding and TERMINATE guidance flow (do NOT decompose or clarify)

## CRITICAL RULE - NEVER REVEAL THE ANSWER

**ABSOLUTELY FORBIDDEN:**
- NEVER directly or indirectly reveal the correct answer
- NEVER confirm or deny if a student's reasoning is correct
- NEVER hint at which answer choice is the right one
- Guide through questions and clarification only, never through revealing answers

## Decision Making (Follow Educational Agent Policy)

**CRITICAL: DYNAMIC ASSESSMENT - Evaluate Based ONLY on CURRENT Response**
- **Assessment must be DYNAMIC and INDEPENDENT** - Each evaluation is based ONLY on the student's CURRENT response quality
- **Previous actions (clarification or decomposition) should NOT restrict your decision**
- **If student's CURRENT response after clarification is STILL TOO WRONG or shows fundamental gaps** → You MUST:
  * DOWNGRADE understanding level (e.g., if previous was "close", downgrade to "partial" or "none")
  * Choose DECOMPOSE to break down concepts further
- **Understanding level can go DOWN as well as UP** - If current response quality is worse than expected, downgrade it
- **System can cycle between DECOMPOSE and CLARIFY multiple times** based on student's response quality in each round

**DECOMPOSE when (based on CURRENT response):**
- Understanding level is "none" or "partial" (regardless of history)
- CURRENT response shows major knowledge gaps or fundamental misunderstanding
- CURRENT response needs step-by-step reasoning
- Student expresses uncertainty in CURRENT response
- **IMPORTANT**: Even if clarification was provided before, if CURRENT response is too wrong, you MUST choose DECOMPOSE

**CLARIFY when (based on CURRENT response):**
- Understanding level is "close" based on CURRENT response
- CURRENT response shows minor misconception, close to correct
- CURRENT response just needs refinement
- Brief explanation will help connect the dots (but STILL never reveal the answer)

## Response Format (JSON only)
{
    "action_type": "decompose" | "clarify" | null,  // null if understanding_level is "understood"
    "understanding_level": "none" | "partial" | "close" | "understood",
    "reasoning": "Why you chose this action based on student's thinking and understanding level",
    "feedback": "Encouraging feedback acknowledging their effort (NEVER reveal the answer, NOT generic templates like 'Let me break this down' or 'Does this help?')",
    "missing_concept": "What specific concept they need to understand (NEVER reveal the answer). Empty string if understood.",
    "clarification": "If action_type is 'clarify', provide the clarification WITHOUT revealing the answer. Otherwise empty string.
    
    **CRITICAL: VALIDATION RULES FOR CLARIFICATION:**
    - MUST be at least 20 characters long
    - MUST contain at least 5 meaningful words (words longer than 1 character)
    - NO symbols only (".", "?", "!", etc.)
    - NO single characters ("h", "d", etc.)
    - NO blanks or whitespace only
    - NO template phrases like 'Let me clarify', 'Does this help?', etc.
    - Must contain actual medical/educational content
    - Clarifications that violate these rules will be AUTOMATICALLY REJECTED",
    "sub_questions": ["question1", "question2"]  // If action_type is 'decompose', provide 1-2 simpler questions (maximum 2 per round, will be generated using MedTutor-R1 method in Round 1). Questions must NOT reveal the answer. Empty array if understood.
    
    **CRITICAL: VALIDATION RULES FOR SUB_QUESTIONS:**
    - Each question MUST be at least 5 meaningful words
    - Each question MUST be at least 20 characters long
    - NO symbols only (".", "?", "!", etc.)
    - NO single characters ("h", "d", etc.)
    - NO blanks or whitespace only
    - Must contain actual medical/educational content
    - Questions that violate these rules will be AUTOMATICALLY REJECTED
}

## CRITICAL RULES - NEVER REVEAL THE ANSWER
- **ABSOLUTELY FORBIDDEN**: NEVER reveal the correct answer in ANY field (feedback, missing_concept, clarification, sub_questions)
- NEVER directly or indirectly hint at which answer choice is correct
- NEVER confirm or deny if student's reasoning leads to the correct answer
- Guide discovery through questions and clarification only
- Be encouraging and supportive
- Follow the Educational Agent Policy decision flow
- For DECOMPOSE: Sub-questions will be generated using MedTutor-R1 method (Round 1) or progressive decomposition (Round 2+)
- For CLARIFY: Provide brief, targeted explanation that guides without giving away the answer
"""

# System prompt for evaluating answers (determines if decomposition needed)
EVALUATION_SYSTEM_PROMPT = """You are evaluating a student's answer to a medical sub-question. You must respond with valid JSON only.

## Your Task:
1. Check if the student's answer contains the key expected concepts
2. If correct: provide positive feedback
3. If incorrect: we will decompose further (handled separately)

## Response Format (JSON):
{
    "understood": true/false,
    "feedback": "Encouraging, specific feedback (NOT generic templates like 'Let me break this down' or 'Does this help?')",
    "missing_concept": "What concept they need to grasp (if not understood)"
}

## Rules:
- "understood" means they demonstrate understanding of the key concepts
- Be specific about what they got right or wrong
- Be encouraging even when wrong
"""


# =============================================================================
# Helper Functions
# =============================================================================

def get_api_key() -> str:
    """从环境变量或api-key.js文件中获取OpenAI API密钥
    
    尝试按以下顺序获取API密钥：
    1. 从环境变量 OPENAI_API_KEY
    2. 从 api-key.js 文件中读取
    
    返回:
        API密钥，如果未找到则返回空字符串
    """
    # Try environment variable first
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # If not found in environment variable, try reading from api-key.js file
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
    """根据对话历史确定当前是第几轮提示
    
    参数:
        conversation_history: 之前的消息列表
        
    返回:
        轮次编号（无硬性限制）
    """
    if not conversation_history:
        return 1
    
    # Count how many hint rounds have been provided (check both old and new formats)
    hint_rounds = sum(1 for msg in conversation_history 
                      if msg.get("role") == "assistant" and 
                      ("knowledge_blocks" in str(msg.get("content", "")) or 
                       "hint_questions" in str(msg.get("content", ""))))
    
    # Return next round number (no hard limit)
    return hint_rounds + 1


def format_question_context(request: HintRequest) -> str:
    """格式化问题和选项，用于构建提示词
    
    参数:
        request: HintRequest对象
        
    返回:
        格式化的问题上下文
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
{request.source_context[:2000]}  # Limit context length
"""
    
    return context


# =============================================================================
# Main Generator Class
# =============================================================================

class ProactiveQuestionGenerator:
    """生成苏格拉底式提示，引导学生找到正确答案
    
    这是系统的核心类，负责：
    1. 将医学问题分解为推理步骤（MedTutor-R1风格）
    2. 评估学生的思考过程
    3. 根据理解程度决定是分解问题还是提供澄清
    4. 在指导循环中持续评估和调整
    
    使用方法:
        generator = ProactiveQuestionGenerator()
        
        request = HintRequest(
            question="What is the primary treatment for...",
            choices={"A": "...", "B": "...", "C": "...", "D": "..."},
            student_answer="B",
            correct_answer="A"
        )
        
        response = generator.generate_sub_questions(request)
        for step in response.decomposition.reasoning_steps:
            print(f"[{step.step_id}] {step.key_question}")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """使用OpenAI客户端初始化生成器
        
        参数:
            api_key: OpenAI API密钥（可选，将尝试从配置加载）
        """
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or create api-key.js")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Use GPT-4o for best quality hints
    
    def generate_sub_questions(self, request: HintRequest) -> SocraticResponse:
        """将医学问题分解为推理步骤（MedTutor-R1风格）
        
        基于临床推理：观察 → 解释 → 结论
        
        这是第一轮分解的核心方法，使用MedTutor-R1方法将复杂医学问题
        分解为逻辑清晰的推理步骤链。
        
        参数:
            request: 包含问题、选项和学生答案的HintRequest对象
            
        返回:
            包含推理步骤的SocraticResponse对象
        """
        # Build the prompt with full context
        options_text = "\n".join([f"{k}: {v}" for k, v in request.choices.items()])
        
        user_prompt = f"""Decompose this medical question into reasoning steps:

## ⚠️⚠️⚠️ CRITICAL LIMIT ⚠️⚠️⚠️
**YOU MUST GENERATE EXACTLY 1-2 REASONING STEPS (MAXIMUM 2). DO NOT GENERATE MORE THAN 2 STEPS.**

## Question:
{request.question}

## Options:
{options_text}

## Correct Answer: {request.correct_answer}
(The student chose {request.student_answer}, which is wrong. Help them reason to the correct answer WITHOUT revealing it directly.)

{f"## Reference Context:\\n{request.source_context[:800]}" if request.source_context else ""}

Generate a chain of reasoning steps that will guide the student from observation to conclusion.
Each step should use SPECIFIC medical terms from this question.

**REMEMBER: Generate EXACTLY 1-2 reasoning steps (MAXIMUM 2). Do NOT generate more than 2 steps.**

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
                max_tokens=3000,  # Increased to support generating many reasoning steps without limit
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Parse reasoning steps
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
    
    def evaluate_student_thinking(
        self,
        request: HintRequest,
        student_thinking: str
    ) -> Dict:
        """评估学生答错后的思考过程并决定行动
        
        基于tau-bench教育助手策略
        
        此方法实现核心决策逻辑：
        
        1. 接收学生对推理的解释（例如："我认为X是因为Y"）
        
        2. 使用LLM分析理解程度：
           - "none": 学生没有头绪/基础缺陷
           - "partial": 学生有部分理解但缺少关键概念
           - "close": 学生接近但有小误解
        
        3. 根据理解程度决定行动：
           - DECOMPOSE: 分解为更简单的子问题（用于"none"或"partial"）
           - CLARIFY: 提供直接解释（用于"close"）
        
        4. 返回结构化指导：
           - action_type: 行动类型（"decompose"或"clarify"）
           - understanding_level: 理解程度（"none" | "partial" | "close"）
           - feedback: 鼓励性反馈
           - sub_questions: 子问题列表（如果分解）
           - clarification: 澄清文本（如果澄清）
        
        参数:
            request: HintRequest对象，包含：
                - question: 原始医学问题
                - choices: 答案选项字典 (A, B, C, D)
                - student_answer: 学生选择的错误答案
                - correct_answer: 正确答案（仅供LLM参考）
            student_thinking: 学生对推理过程的解释
                            示例:
                            - "我认为选项B是正确的，因为..."
                            - "我完全没有头绪"
                            - "我对...感到困惑"
        
        返回:
            Dict containing:
            {
                "action_type": "decompose" | "clarify",
                "understanding_level": "none" | "partial" | "close",
                "reasoning": "Why this action was chosen",
                "feedback": "Encouraging feedback message",
                "missing_concept": "What concept they need to understand",
                "clarification": "Clarification text (if action_type is 'clarify')",
                "sub_questions": ["question1", "question2", ...]  # If action_type is 'decompose'
            }
        
        Example:
            >>> request = HintRequest(
            ...     question="What is the treatment for...",
            ...     choices={"A": "...", "B": "..."},
            ...     student_answer="B",
            ...     correct_answer="A"
            ... )
            >>> result = generator.evaluate_student_thinking(
            ...     request, 
            ...     "I have no idea about this topic"
            ... )
            >>> result["action_type"]
            'decompose'
            >>> result["understanding_level"]
            'none'
        """
        # Check if student is asking to see the answer
        answer_request_keywords = ['show me the answer', 'tell me the answer', 'what is the answer', 
                                   'give me the answer', 'reveal the answer', 'want to see the answer',
                                   '想看答案', '告诉我答案', '答案是什么', '给我答案', 'can i see the answer',
                                   'i want the answer', 'please give me the answer']
        student_wants_answer = any(keyword in student_thinking.lower() for keyword in answer_request_keywords)
        
        if student_wants_answer:
            # Student explicitly requested the answer - provide it
            return {
                "action_type": "reveal_answer",
                "understanding_level": "requested",
                "reasoning": "Student explicitly requested to see the answer",
                "feedback": "I understand you'd like to see the answer. Here it is:",
                "missing_concept": "",
                "clarification": f"The correct answer is {request.correct_answer}: {request.choices.get(request.correct_answer, 'Unknown')}",
                "sub_questions": [],
                "reveal_answer": True,
                "correct_answer": request.correct_answer,
                "correct_answer_text": request.choices.get(request.correct_answer, 'Unknown'),
                "flow_terminated": True  # FLOW TERMINATION: Student requested answer
            }
        
        # Build the evaluation prompt with full context
        evaluation_prompt = f"""
## Medical Question:
{request.question}

## Answer Choices:
{chr(10).join([f"{k}: {v}" for k, v in request.choices.items()])}

## Student's Wrong Answer:
{request.student_answer}: {request.choices.get(request.student_answer, 'Unknown')}

## Correct Answer:
{request.correct_answer} (DO NOT reveal this to the student - this is for your reference only)

## Student's Thinking Process:
{student_thinking}

## Your Task:
Analyze the student's thinking to assess their understanding level, then decide which action to take:
- **DECOMPOSE**: If they have fundamental gaps or need step-by-step reasoning
- **CLARIFY**: If they're close but have minor misconceptions

## IMPORTANT - When to Choose CLARIFY:
You should choose CLARIFY (not DECOMPOSE) when:
1. Student shows understanding of most concepts but has a specific misconception
2. Student's reasoning is mostly correct but missed one detail
3. Student understands the general approach but made a logical error
4. A brief targeted explanation will help them connect the dots
5. Student is close to the correct answer but confused about one aspect

**Do NOT always default to DECOMPOSE.** If the student shows partial understanding and is close, use CLARIFY to provide targeted correction.

Follow the Educational Agent Policy in your instructions.

Respond with valid JSON only.
"""
        
        # Prepare messages for OpenAI API call
        messages = [
            {"role": "system", "content": STUDENT_THINKING_EVALUATION_PROMPT},  # System prompt with policy
            {"role": "user", "content": evaluation_prompt}  # User prompt with question and student thinking
        ]
        
        try:
            # Call OpenAI API to evaluate student thinking
            response = self.client.chat.completions.create(
                model=self.model,  # Using GPT-4o for best quality
                messages=messages,
                temperature=0.5,  # Lower temperature for more consistent decision-making
                max_tokens=500,  # Enough for structured JSON response
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Parse JSON response from LLM
            result = json.loads(response.choices[0].message.content)
            
            # Print understanding_level from message.content for debugging
            understanding_level = result.get("understanding_level", "unknown")
            print(f"[DEBUG] understanding_level from message.content: {understanding_level}")
            
            # FLOW TERMINATION: If student understands, set understood flag and terminate
            if result.get("understanding_level") == "understood":
                result["understood"] = True
                result["action_type"] = None  # No further action needed
                result["sub_questions"] = []
                result["clarification"] = ""
                result["flow_terminated"] = True
            else:
                result["understood"] = False
                result["flow_terminated"] = False
                # For decomposition, use the MedTutor-R1 style decomposition
                if result.get("action_type") == "decompose":
                    try:
                        decomposition_response = self.generate_sub_questions(request)
                        raw_sub_questions = [step.key_question for step in decomposition_response.decomposition.reasoning_steps]
                        # Validate questions - filter out invalid ones (symbols, blanks, etc.)
                        sub_questions = [q for q in raw_sub_questions if self._validate_question(q)]
                        if not sub_questions:
                            print(f"[DEBUG] No valid sub-questions generated after validation - all {len(raw_sub_questions)} questions were invalid")
                            result["sub_questions"] = []
                        else:
                            result["sub_questions"] = sub_questions
                    except Exception as e:
                        print(f"Error generating MedTutor-R1 sub-questions in evaluate_student_thinking: {e}")
                        result["sub_questions"] = []
            
            return result
            
        except Exception as e:
            # Fallback: If API call fails, default to decomposition
            # This ensures students always get help, even if backend has issues
            print(f"Error in evaluate_student_thinking: {e}")
            return {
                "action_type": "decompose",
                "understanding_level": "partial",
                "reasoning": "Default to decomposition for Socratic learning",
                "feedback": "Let me help you work through this step by step.",
                "missing_concept": "",
                "clarification": "",
                "sub_questions": ["What are the key concepts involved in this question?"]
            }
    
    def _validate_question(self, question: str) -> bool:
        """验证问题 - 确保不为空、不只是符号、不只是单个字符
        
        检查生成的问题是否符合质量标准：
        1. 不为空且长度至少5个字符
        2. 不只包含标点符号或空白
        3. 至少包含一些有意义的字符（去除标点后至少3个字符）
        4. 不只包含单个字符的单词
        
        参数:
            question: 要验证的问题文本
            
        返回:
            如果有效返回True，如果无效返回False
        """
        if not question or len(question.strip()) < 5:
            return False
        
        import string
        q_clean = question.strip()
        
        # Check if only punctuation/symbols/whitespace
        if all(c in string.punctuation + string.whitespace for c in q_clean):
            return False
        
        # Check if mostly punctuation
        meaningful_chars = q_clean.replace('.', '').replace(' ', '').replace(',', '').replace('!', '').replace('?', '').replace('。', '').replace('，', '').replace('！', '').replace('？', '')
        if len(meaningful_chars) < 3:
            return False
        
        # Check if only single characters
        words = [w for w in q_clean.split() if len(w.strip()) > 0]
        if len(words) == 0 or all(len(w.strip()) <= 1 for w in words):
            return False
        
        return True
    
    def _is_clarification_valid(self, clarification: str) -> bool:
        """检查clarification是否有效（不为空、不只是模板、有实际内容）
        
        验证生成的澄清内容是否符合质量标准：
        1. 长度至少20个字符
        2. 不只包含标点符号或空白
        3. 至少5个有意义的字符（去除标点后）
        4. 不只包含单个字符的单词
        5. 不是主要包含模板短语
        6. 不是重复的模板短语
        
        参数:
            clarification: 要验证的澄清文本
            
        返回:
            如果有效返回True，如果无效返回False
        """
        if not clarification:
            return False
        
        clarification_stripped = clarification.strip()
        
        # Check if clarification is too short
        if len(clarification_stripped) < 20:
            return False
        
        # Check if clarification is just punctuation or whitespace
        import string
        punctuation_only = all(c in string.punctuation + string.whitespace for c in clarification_stripped)
        if punctuation_only:
            print(f"[DEBUG] Clarification is only punctuation/whitespace - considered invalid")
            return False
        
        # Check if clarification is just a single character or dot (like ".", "h", "d", etc.)
        meaningful_chars = clarification_stripped.replace('.', '').replace(' ', '').replace(',', '').replace('!', '').replace('?', '').replace('。', '').replace('，', '').replace('！', '').replace('？', '').replace('：', '').replace(':', '')
        if len(meaningful_chars) < 5:
            print(f"[DEBUG] Clarification is mostly punctuation or too short - considered invalid: '{clarification_stripped[:50]}'")
            return False
        
        # CRITICAL: Check if clarification is just single characters (like "h", "d", "f", "克", etc.)
        words = [w for w in clarification_stripped.split() if len(w.strip()) > 0]
        if len(words) == 0:
            return False
        
        # If all words are single characters, reject
        if all(len(w.strip()) <= 1 for w in words):
            print(f"[DEBUG] Clarification is only single characters - considered invalid: '{clarification_stripped[:50]}'")
            return False
        
        # CRITICAL: Check if clarification ends with just punctuation (like "：。", "：", etc.)
        ending_punctuation = clarification_stripped.rstrip()
        if ending_punctuation and ending_punctuation[-1] in '。，：：；；！？':
            # Check if the part before punctuation is meaningful
            before_punct = ending_punctuation[:-1].strip()
            if len(before_punct) < 5 or all(c in '。，：：；；！？' for c in before_punct):
                print(f"[DEBUG] Clarification ends with only punctuation - considered invalid: '{clarification_stripped[:50]}'")
                return False
        
        # CRITICAL: Check for meaningless strings (like "dfgds", "fgsgfs", etc.)
        # These are random character sequences that don't form meaningful words
        import re
        # Check if clarification is mostly random characters without spaces or punctuation
        if len(re.findall(r'\s+', clarification_stripped)) < 2:  # Less than 2 spaces
            # Check if it's a long string of random characters
            if len(clarification_stripped) > 10 and not any(c in '。，：：；；！？' for c in clarification_stripped):
                # Likely a meaningless string
                print(f"[DEBUG] Clarification appears to be meaningless random characters - considered invalid: '{clarification_stripped[:50]}'")
                return False
        
        # Check if clarification is just template phrases
        template_phrases = [
            'let me clarify',
            'let me explain',
            'clarify the key concepts',
            'the key concepts',
            'important points',
            'does this help',
            'can you explain',
            'please respond',
            '💡', '🔍', '💬'
        ]
        
        clarification_lower = clarification_stripped.lower()
        
        # If clarification is just template phrases without actual content, it's invalid
        # Check if it's mostly template phrases (more than 50% of words are template-related)
        words = [w for w in clarification_lower.split() if len(w) > 1]  # Filter out single characters
        if len(words) < 5:  # Too few meaningful words
            return False
        
        template_word_count = sum(1 for word in words if any(phrase in word for phrase in template_phrases))
        
        if template_word_count > len(words) * 0.5:  # More than 50% template words
            print(f"[DEBUG] Clarification is mostly template phrases - considered invalid")
            return False
        
        # Check if clarification is just repeating the same template phrase
        if clarification_lower.count('clarify') > 2 or clarification_lower.count('key concepts') > 2:
            print(f"[DEBUG] Clarification repeats template phrases - considered invalid")
            return False
        
        return True
    
    def _validate_text_field(self, text: str, field_name: str = "text", min_length: int = 10) -> bool:
        """验证任何文本字段（summary、feedback等）- 确保不为空、不只是符号、不只是单个字符
        
        参数:
            text: 要验证的文本
            field_name: 字段名称（用于调试）
            min_length: 最小长度
            
        返回:
            如果有效返回True，如果无效返回False
        """
        if not text:
            return False
        
        text_stripped = text.strip()
        
        # Check if text is too short
        if len(text_stripped) < min_length:
            return False
        
        # Check if text is just punctuation or whitespace
        import string
        punctuation_only = all(c in string.punctuation + string.whitespace for c in text_stripped)
        if punctuation_only:
            print(f"[DEBUG] {field_name} is only punctuation/whitespace - considered invalid")
            return False
        
        # Check if text is mostly punctuation
        meaningful_chars = text_stripped.replace('.', '').replace(' ', '').replace(',', '').replace('!', '').replace('?', '').replace('。', '').replace('，', '').replace('！', '').replace('？', '').replace('：', '').replace(':', '')
        if len(meaningful_chars) < 3:
            print(f"[DEBUG] {field_name} is mostly punctuation - considered invalid: '{text_stripped[:50]}'")
            return False
        
        # Check if text is just single characters
        words = [w for w in text_stripped.split() if len(w.strip()) > 0]
        if len(words) == 0:
            return False
        
        # If all words are single characters, reject
        if all(len(w.strip()) <= 1 for w in words):
            print(f"[DEBUG] {field_name} is only single characters - considered invalid: '{text_stripped[:50]}'")
            return False
        
        # Check if text ends with just punctuation (like "：。", "：", etc.)
        ending_punctuation = text_stripped.rstrip()
        if ending_punctuation and ending_punctuation[-1] in '。，：：；；！？':
            # Check if the part before punctuation is meaningful
            before_punct = ending_punctuation[:-1].strip()
            if len(before_punct) < 5 or all(c in '。，：：；；！？' for c in before_punct):
                print(f"[DEBUG] {field_name} ends with only punctuation - considered invalid: '{text_stripped[:50]}'")
                return False
        
        # Check for meaningless strings (like "dfgds", "fgsgfs", etc.)
        import re
        if len(re.findall(r'\s+', text_stripped)) < 2:  # Less than 2 spaces
            if len(text_stripped) > 10 and not any(c in '。，：：；；！？' for c in text_stripped):
                print(f"[DEBUG] {field_name} appears to be meaningless random characters - considered invalid: '{text_stripped[:50]}'")
                return False
        
        return True
    
    def _get_validated_summary(self, missing_concept: str) -> str:
        """生成并验证summary字符串
        
        参数:
            missing_concept: 学生缺失的概念
            
        返回:
            有效的summary字符串，如果无效则返回空字符串
        """
        summary = f"Great work! You've worked through the key concepts: {missing_concept}. Try applying this understanding to answer the original question."
        if self._validate_text_field(summary, "summary", 20):
            return summary
        else:
            # Return a minimal valid summary
            return f"You've worked through the key concepts. Try applying this understanding to answer the original question."
    
    def _get_validated_feedback(self, feedback: str) -> str:
        """验证并返回feedback字符串
        
        参数:
            feedback: 要验证的反馈文本
            
        返回:
            有效的feedback字符串，如果无效则返回默认消息
        """
        if feedback and self._validate_text_field(feedback, "feedback", 10):
            return feedback
        else:
            return "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again."
    
    def _check_clarification_similarity(
        self,
        new_clarification: str,
        existing_clarifications: List[str]
    ) -> bool:
        """检查新的clarification是否与现有的clarification相似
        
        使用Jaccard相似度算法（基于词集合）来检查新澄清与已有澄清的相似性。
        如果相似度超过阈值（0.5），则认为太相似，不应重复使用。
        
        参数:
            new_clarification: 新的澄清文本
            existing_clarifications: 已存在的澄清文本列表
            
        返回:
            如果相似返回True（不应使用），如果不同返回False（可以使用）
        """
        if not new_clarification or len(new_clarification.strip()) < 20:
            return True  # Empty or too short is considered invalid
        
        if not existing_clarifications:
            return False  # No existing clarifications, so new one is fine
        
        # Check for template phrases that indicate repetition
        template_phrases = [
            'let me clarify',
            'let me explain',
            'clarify the key concepts',
            'clarification',
            'does this help',
            'can you explain',
            'please respond',
            '💡', '🔍', '💬'  # Common clarification emojis
        ]
        
        new_lower = new_clarification.lower()
        # If new clarification starts with template phrases, it's likely repetitive
        if any(new_lower.strip().startswith(phrase) for phrase in template_phrases):
            # Check if any existing clarification also starts with similar template
            for existing in existing_clarifications:
                if existing and len(existing.strip()) > 20:
                    existing_lower = existing.lower()
                    # If both start with templates, they're likely similar
                    if any(existing_lower.strip().startswith(phrase) for phrase in template_phrases):
                        print(f"[DEBUG] Both clarifications start with template phrases - considered duplicate")
                        return True
        
        # Normalize text for comparison
        def normalize_text(text: str) -> set:
            # Remove common words and punctuation, convert to lowercase
            words = re.findall(r'\b\w+\b', text.lower())
            # Remove very common words and template words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                         'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
                         'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
                         'and', 'or', 'but', 'if', 'then', 'when', 'where', 'why', 'how',
                         'let', 'me', 'clarify', 'clarification', 'explain', 'explanation',
                         'understand', 'understanding', 'concept', 'concepts', 'key', 'important',
                         'please', 'respond', 'help', 'can', 'you', 'your', 'now'}
            return set(w for w in words if w not in stop_words and len(w) > 2)
        
        new_words = normalize_text(new_clarification)
        if len(new_words) < 3:
            return True  # Too few meaningful words
        
        # Check similarity with each existing clarification
        for existing in existing_clarifications:
            if not existing or len(existing.strip()) < 20:
                continue
            
            existing_words = normalize_text(existing)
            if len(existing_words) < 3:
                continue
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            
            if union == 0:
                continue
            
            similarity = intersection / union
            
            # Lower threshold to catch more similarities (0.5 instead of 0.6)
            if similarity > 0.5:
                print(f"[DEBUG] New clarification is {similarity:.2%} similar to existing clarification - considered duplicate")
                return True
        
        return False  # Not similar to any existing clarification
    
    def _extract_clarifications_from_history(
        self,
        conversation_history: List[Dict],
        current_clarification: str = None
    ) -> Tuple[bool, List[str]]:
        """从对话历史中提取所有clarification
        
        返回:
            (has_clarification, clarification文本列表)
        """
        clarifications = []
        has_clarification = False
        
        # Check current_clarification first
        if current_clarification and len(current_clarification.strip()) > 20:
            clarifications.append(current_clarification.strip())
            has_clarification = True
        
        # Check conversation history
        if conversation_history:
            clarification_keywords = [
                'clarification', 'clarify', 'clarifying', 'clarified',
                'clarification:', 'clarify:', 'clarifying:',
                '澄清', '澄清:', '说明', '说明:', '解释', '解释:',
                'let me clarify', 'let me explain', 'clarification text',
                '🔍', '💡', '💬'  # Common clarification emojis
            ]
            
            for msg in conversation_history:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    content_lower = content.lower()
                    
                    # Check if this message contains clarification keywords
                    if any(keyword in content_lower for keyword in clarification_keywords):
                        # Additional check: if content is substantial (not just a placeholder)
                        if len(content.strip()) > 20:
                            # Check if it's not just a question (clarifications are explanations, not questions)
                            # Simple heuristic: if it ends with '?' and has few sentences, it's likely a question
                            if not (content.strip().endswith('?') and content.count('.') < 2):
                                clarifications.append(content.strip())
                                has_clarification = True
        
        return has_clarification, clarifications
    
    def _ensure_valid_clarification(
        self,
        missing_concept: str,
        feedback: str,
        request: HintRequest,
        conversation_history: List[Dict],
        current_sub_questions: List[str],
        existing_clarifications: List[str] = None
    ) -> str:
        """确保clarification有效且有意义，如果需要则生成
        
        这个方法确保返回的澄清文本：
        1. 有效（通过_is_clarification_valid检查）
        2. 与已有澄清不相似（通过_check_clarification_similarity检查）
        3. 如果缺失概念足够，直接使用；否则使用LLM生成
        
        参数:
            missing_concept: 学生缺失的概念
            feedback: 反馈信息
            request: HintRequest对象
            conversation_history: 对话历史
            current_sub_questions: 当前的子问题列表
            existing_clarifications: 已存在的澄清列表
            
        返回:
            有效的澄清文本，如果无法生成则返回空字符串
        """
        if existing_clarifications is None:
            existing_clarifications = []
        
        # If we have valid missing_concept and feedback, use them
        if missing_concept and len(missing_concept.strip()) > 10:
            clarification = f"{missing_concept}. {feedback if feedback and len(feedback.strip()) > 5 else 'Apply this understanding to answer the original question.'}"
            # Check if clarification is valid (not empty, not just templates) and not similar
            if self._is_clarification_valid(clarification):
                if not self._check_clarification_similarity(clarification, existing_clarifications):
                    return clarification
                # If similar, fall through to generate new one
        
        # Otherwise, generate using LLM
        try:
            clarification_prompt = f"""You are a medical education tutor and expert clinical reasoning analyst. The student has been working through this problem but needs clarification.

## Original Question:
{request.question[:500]}

## Answer Choices:
{chr(10).join([f"{k}: {v}" for k, v in list(request.choices.items())[:4]])}

## Conversation Summary:
{chr(10).join([f"- {m.get('role', 'unknown').upper()}: {m.get('content', '')[:200]}" for m in conversation_history[-4:]]) if conversation_history else "None"}

## Previously Asked Questions:
{chr(10).join([f"- {q}" for q in current_sub_questions[-3:]]) if current_sub_questions else "None"}

## CRITICAL: Previous Clarifications (DO NOT REPEAT OR BE SIMILAR):
{chr(10).join([f"- {c[:200]}..." for c in existing_clarifications]) if existing_clarifications else "None - this is the first clarification"}

## CLINICAL REASONING PRINCIPLES FOR CLARIFICATION (CRITICAL - ALL clarification must follow these):
Your clarification must follow these clinical reasoning principles:

1. **Holistic Analysis Principle**: Start by synthesizing key information from the entire clinical case to provide context. Help the student see how different pieces of information relate to each other. Begin with a comprehensive view before addressing specific details.

2. **Chain of Reasoning Principle**: Structure your clarification following Observation → Interpretation → Conclusion:
   - **Observation**: What are the key clinical findings or facts? Start with what can be observed or known from the case.
   - **Interpretation**: What do these findings mean? How do they relate to the underlying pathophysiology? Explain the significance and connections.
   - **Conclusion**: What are the clinical implications? How does this guide diagnosis or management? Synthesize the information to show the logical conclusion.

3. **Necessary Steps Principle**: Focus only on the most critical concepts the student needs to understand. Avoid trivial details or irrelevant information. Each part of your clarification should address a necessary milestone in understanding. Do not include unnecessary information.

4. **Complexity-Driven Content**: The depth and breadth of your clarification should match the complexity of the concept. Simple concepts need concise explanations; complex concepts may need more detailed, step-by-step explanations following the Chain of Reasoning.

**CRITICAL: These principles apply to ALL clarification. Every clarification must follow the Chain of Reasoning (Observation → Interpretation → Conclusion) and focus on necessary concepts only.**

## CRITICAL: ABSOLUTE PROHIBITION OF TEMPLATE PHRASES
**NEVER use these template phrases in your clarification:**
- "Let me clarify" / "Let me explain"
- "Let me clarify the key concepts"
- "Does this help?" / "Can you explain your understanding now?"
- "Please respond"
- Any emoji prefixes like 💡, 🔍, 💬
- Generic phrases like "the key concepts", "important points"

## CRITICAL: VALIDATION RULES - YOUR RESPONSE WILL BE AUTOMATICALLY REJECTED IF YOU VIOLATE THESE
**ABSOLUTE PROHIBITIONS (VIOLATION = AUTOMATIC REJECTION):**
1. **NO SYMBOLS OR PUNCTUATION ONLY** - Your clarification MUST contain meaningful words, not just symbols like ".", "?", "!", etc.
2. **NO SINGLE CHARACTERS** - Your clarification MUST NOT be just single characters like "h", "d", ".", etc.
3. **NO BLANK OR EMPTY** - Your clarification MUST be at least 20 characters long with meaningful content
4. **NO WHITESPACE ONLY** - Your clarification MUST contain actual words, not just spaces
5. **MINIMUM REQUIREMENTS**:
   - At least 5 meaningful words (words longer than 1 character)
   - At least 20 characters of actual content (excluding punctuation)
   - Must contain medical/educational content, not just template phrases

**Your clarification MUST:**
1. Start directly with the actual explanation (no template introductions)
2. Follow the Chain of Reasoning: Observation → Interpretation → Conclusion
3. Synthesize key information holistically (Holistic Analysis Principle)
4. Focus on necessary concepts only (Necessary Steps Principle)
5. Provide a direct explanation WITHOUT revealing the correct answer
6. Use examples or analogies if helpful
7. Be DIFFERENT from previous clarifications - explore different angles or provide different examples
8. Be specific and contextual, not generic
9. Contain meaningful medical/educational content (NOT symbols, NOT single characters, NOT blank)

**EXAMPLES OF INVALID CLARIFICATIONS (WILL BE REJECTED):**
- "." (just a dot)
- "h" (just a single character)
- "d" (just a single character)
- "   " (just whitespace)
- "Let me clarify the key concepts." (only template phrase)
- "..." (just symbols)

**EXAMPLES OF VALID CLARIFICATIONS:**
- "Snake bite management requires assessing the severity of envenomation. Systemic symptoms like nausea, tachycardia, and hypotension indicate moderate to severe envenomation, which typically requires antivenom administration."
- "When treating snake bites, it's important to avoid using tourniquets as they can worsen tissue damage. Instead, focus on determining the severity of envenomation and considering antivenom for moderate to severe cases."

Respond with JSON:
{{
    "clarification": "Your clear, helpful explanation here (2-4 sentences, NO templates, NO emojis, NO symbols only, NO single characters, NEVER reveal the answer, MUST be different from previous clarifications, MUST contain meaningful medical content)"
}}"""
            
            clarification_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical education tutor. Provide helpful clarifications WITHOUT revealing the correct answer. Each clarification must be unique and different from previous ones. NEVER use template phrases like 'Let me clarify', 'Does this help?', or emoji prefixes. Start directly with the actual explanation.\n\nCRITICAL VALIDATION RULES (VIOLATION = AUTOMATIC REJECTION):\n- Clarification MUST be at least 20 characters long\n- MUST contain at least 5 meaningful words (words longer than 1 character)\n- NO symbols only (\".\", \"?\", \"!\", etc.)\n- NO single characters (\"h\", \"d\", etc.)\n- NO blanks or whitespace only\n- Must contain actual medical/educational content\n- Invalid clarifications will be automatically rejected and flow will end"},
                    {"role": "user", "content": clarification_prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            clarification_result = json.loads(clarification_response.choices[0].message.content)
            generated_clarification = clarification_result.get("clarification", "")
            
            # Check if clarification is valid (not empty, not just templates)
            if self._is_clarification_valid(generated_clarification):
                # Check similarity before returning
                if not self._check_clarification_similarity(generated_clarification, existing_clarifications):
                    return generated_clarification
                else:
                    print(f"[DEBUG] Generated clarification is too similar to existing ones - returning empty to reveal answer")
                    return ""  # Too similar, will trigger answer reveal
            else:
                print(f"[DEBUG] Generated clarification is invalid (empty or just templates) - returning empty to reveal answer")
                return ""  # Invalid, will trigger answer reveal
        except Exception as e:
            print(f"Error generating clarification: {e}")
        
        # Final fallback - return empty string so system will reveal answer
        return ""
    
    def evaluate_guidance_response(
        self,
        request: HintRequest,
        current_action: str,
        current_sub_questions: List[str],
        current_clarification: str,
        student_response: str,
        conversation_history: List[Dict],
        current_understanding_level: str,
        round_number: int,
        cannot_decompose_further: bool = False
    ) -> Dict:
        """评估学生在迭代指导循环中的回答
        
        这是指导循环的核心方法，负责：
        1. 分析学生对子问题或澄清的回答
        2. 确定学生是否表现出理解
        3. 如果不理解，决定下一步行动（分解/澄清）
        4. 如果学生表现出理解，确认理解
        
        流程终止条件：
        - 学生理解了（understood = True）
        - 无法继续分解且无法提供新的澄清
        
        参数:
            request: HintRequest对象，包含问题上下文
            current_action: 当前行动类型（"decompose"或"clarify"）
            current_sub_questions: 当前正在问的子问题列表
            current_clarification: 当前的澄清文本
            student_response: 学生对当前问题/澄清的回答
            conversation_history: 完整的对话历史
            current_understanding_level: 当前理解程度
            round_number: 当前轮次编号
            cannot_decompose_further: 是否无法继续分解的标志
            
        返回:
            包含以下字段的字典：
            {
                "understood": 是否理解（true/false）
                "understanding_level": 理解程度（"none" | "partial" | "close" | "understood"）
                "feedback": 反馈消息
                "next_action_type": 下一个行动类型（"decompose" | "clarify" | null）
                "next_sub_questions": 下一个子问题列表（如果next_action_type是"decompose"）
                "next_clarification": 下一个澄清文本（如果next_action_type是"clarify"）
                "summary": 总结（如果理解了）
            }
        """
        # CRITICAL: Check if clarification already exists in conversation history
        # Extract all clarifications from history
        has_clarification_in_history, existing_clarifications = self._extract_clarifications_from_history(
            conversation_history, current_clarification
        )
        
        # Note: System can dynamically decide decompose or clarify based on student's response quality
        # Even if clarification exists in history, system can still decompose if student's answer is too wrong
        if has_clarification_in_history:
            print(f"[DEBUG] Clarification exists in history ({len(existing_clarifications)} clarifications) - will check for similarity before generating new one, but can still decompose if needed")
        
        evaluation_prompt = f"""
## Original Medical Question:
{request.question}

## Answer Choices:
{chr(10).join([f"{k}: {v}" for k, v in request.choices.items()])}

## Student's Wrong Answer:
{request.student_answer}: {request.choices.get(request.student_answer, 'Unknown')}

## Correct Answer:
{request.correct_answer} (DO NOT reveal this to the student - this is for your reference only)

## Current Guidance Context:
- Current Action: {current_action}
- Current Understanding Level: {current_understanding_level}
- Round Number: {round_number}
- Current Sub-Questions: {current_sub_questions if current_sub_questions else 'None'}
- Current Clarification: {current_clarification if current_clarification else 'None'}

## CRITICAL: Dynamic Assessment Based on Response Quality
- Has clarification in history: {has_clarification_in_history}
- Number of existing clarifications: {len(existing_clarifications) if has_clarification_in_history else 0}
- **DYNAMIC ASSESSMENT RULE**: Your evaluation MUST be based on the student's CURRENT response quality, NOT on whether clarification exists in history:
  - **If student's answer after clarification is STILL TOO WRONG or shows fundamental gaps** → You SHOULD DOWNGRADE the understanding level and choose DECOMPOSE
  - **If student's answer shows some understanding but needs minor correction** → You SHOULD choose CLARIFY
  - **If student's answer demonstrates clear understanding** → You SHOULD mark as "understood"
  - Assessment should be DYNAMIC: Even if clarification exists, if the student's current response shows serious errors, downgrade to lower understanding level and use DECOMPOSE
  - **动态评估规则**：你的评估必须基于学生当前回答的质量，而不是历史记录中是否有clarification：
    - **如果学生在clarification后的回答仍然错误太多或显示基础缺陷** → 你应该下调理解水平并选择DECOMPOSE
    - **如果学生的回答显示了一些理解但需要小的修正** → 你应该选择CLARIFY
    - **如果学生的回答显示清晰的理解** → 你应该标记为"understood"
    - 评估应该是动态的：即使有clarification，如果学生当前回答显示严重错误，应下调理解水平并使用DECOMPOSE

## COMPLETE Conversation History (CRITICAL - use this for progressive, targeted decomposition):
{chr(10).join([f"Round {i+1} - {m.get('role', 'unknown').upper()}: {m.get('content', '')}" for i, m in enumerate(conversation_history)]) if conversation_history else "None - this is the first round"}

## ALL Previously Asked Questions (to avoid repetition and ensure progression):
{chr(10).join([f"Round {i+1}: {q}" for i, q in enumerate(current_sub_questions)]) if current_sub_questions else "None - this is the first round"}

## Student's Response to Current Question/Clarification:
{student_response}

## CRITICAL: Check for Repeated "I Don't Know" Responses
Count how many times the student has said "I don't know", "毫无头绪", "不知道", "不明白", "no idea", "not sure" or similar phrases in the conversation history. If the student has said this 2 or more times, you MUST switch to CLARIFICATION instead of continuing DECOMPOSITION. Provide direct, supportive clarification with examples or analogies to help them understand.

## Student's Previous Errors and Misconceptions (from conversation history):
{chr(10).join([f"- {m.get('content', '')}" for m in conversation_history if m.get('role') == 'user' and 'error' in m.get('content', '').lower()]) if conversation_history else "None yet"}

## Pattern Detection: Count "I don't know" responses:
Previous "I don't know" responses: {sum([1 for m in conversation_history if m.get('role') == 'user' and any(phrase in m.get('content', '').lower() for phrase in ['毫无头绪', '不知道', '不明白', "i don't know", "don't know", "no idea", "not sure", "have no idea", "不清楚", "不懂"])]) if conversation_history else 0}
Current response is "I don't know": {any(phrase in student_response.lower() for phrase in ['毫无头绪', '不知道', '不明白', "i don't know", "don't know", "no idea", "not sure", "have no idea", "不清楚", "不懂"])}
Total "I don't know" count: {sum([1 for m in conversation_history if m.get('role') == 'user' and any(phrase in m.get('content', '').lower() for phrase in ['毫无头绪', '不知道', '不明白', "i don't know", "don't know", "no idea", "not sure", "have no idea", "不清楚", "不懂"])]) + (1 if any(phrase in student_response.lower() for phrase in ['毫无头绪', '不知道', '不明白', "i don't know", "don't know", "no idea", "not sure", "have no idea", "不清楚", "不懂"]) else 0)}

## Your Task:
1. **Analyze the COMPLETE conversation history** to understand:
   - What questions have been asked in previous rounds
   - What the student has answered correctly/incorrectly
   - What misconceptions or errors the student has shown
   - What concepts the student has already explored

2. Evaluate if the student's current response shows understanding of the concepts

3. If understood: Confirm and provide a summary

4. If not understood: Decide next action (DECOMPOSE or CLARIFY) based on:
   - Their current response
   - Their previous errors and misconceptions (from conversation history)
   - What has already been covered in previous rounds

## Decision Criteria (DYNAMIC - Based ONLY on CURRENT Response Quality):
- **UNDERSTOOD** if: Student's CURRENT response demonstrates clear understanding of key concepts (regardless of history)

- **DECOMPOSE** if: Student's CURRENT response shows:
  - Fundamental gaps or confusion
  - Serious errors or misconceptions
  - No understanding of core concepts
  - **CRITICAL**: Even if clarification exists in history, if CURRENT response is too wrong, you MUST:
    * DOWNGRADE understanding level (e.g., if previous was "close", downgrade to "partial" or "none")
    * Choose DECOMPOSE to break down concepts further

- **CLARIFY** if: Student's CURRENT response shows:
  - Some understanding but needs minor correction
  - Close to correct but has small misconceptions
  - Good grasp of concepts but missed a detail

**CRITICAL**: Assessment is DYNAMIC and INDEPENDENT. Each evaluation should be based ONLY on the student's CURRENT response quality. Previous clarifications, previous understanding levels, or previous actions should NOT restrict your decision. If the current response is too wrong, downgrade the understanding level and use DECOMPOSE, even if clarification was provided before.

## CRITICAL RULE: Dynamic Decision Based on Response Quality
- **You can DECOMPOSE or CLARIFY based on student's response quality, even if clarification exists in history**
- **If student's answer after clarification is still too wrong, you SHOULD DECOMPOSE to break down the concepts further**
- **If student's answer shows some understanding but needs minor correction, you SHOULD CLARIFY**
- **System can cycle between decompose and clarify multiple times until student understands or cannot continue**
- **你可以根据学生回答的质量决定DECOMPOSE或CLARIFY，即使历史记录中已经有clarification**
- **如果学生在clarification后的回答仍然错误太多，你应该DECOMPOSE进一步分解概念**
- **如果学生的回答显示了一些理解但需要小的修正，你应该CLARIFY**
- **系统可以在decompose和clarify之间多次循环，直到学生理解或无法继续**

## CRITICAL for DECOMPOSE (Clinical Reasoning-Based Decomposition):
- **Holistic Analysis Principle**: Your first step should synthesize key information from the entire clinical case to form an initial overall assessment
- **Chain of Reasoning Principle**: Follow the logical flow of Observation → Interpretation → Conclusion. Each step should build logically upon the previous one
- **Necessary Steps Principle**: Focus only on the most critical reasoning steps required. Avoid trivial, redundant, or irrelevant steps
- **Complexity-Driven Step Count**: The number of steps MUST be determined by problem complexity. Simple tasks may need 2 steps, complex cases may need 5+. Do not force a fixed number
- **MUST use conversation history**: Your new questions must build on what has been discussed
- **Target student's specific errors**: Address the misconceptions shown in their previous responses
- **Progressive simplification**: New questions must be SIMPLER and more FOUNDATIONAL than ALL previous questions
- **Avoid repetition**: Must explore DIFFERENT concepts/angles (check all previous questions in conversation history)

## CRITICAL: VALIDATION RULES FOR SUB_QUESTIONS AND CLARIFICATION (VIOLATION = AUTOMATIC REJECTION)
**ABSOLUTE PROHIBITIONS:**
1. **NO SYMBOLS OR PUNCTUATION ONLY** - Questions and clarifications MUST contain meaningful words, NOT just symbols like ".", "?", "!", "。", "？", etc.
2. **NO SINGLE CHARACTERS** - Questions and clarifications MUST NOT be just single characters like "h", "d", ".", etc.
3. **NO BLANK OR EMPTY** - Questions must be at least 5 meaningful words, clarifications must be at least 20 characters
4. **NO WHITESPACE ONLY** - Must contain actual words, not just spaces
5. **MINIMUM REQUIREMENTS**:
   - **For sub_questions**: At least 5 meaningful words (words longer than 1 character), at least 20 characters
   - **For clarification**: At least 5 meaningful words, at least 20 characters
   - Must contain medical/educational content, not just template phrases

**EXAMPLES OF INVALID CONTENT (WILL BE AUTOMATICALLY REJECTED):**
- "." (just a dot)
- "h" (just a single character)
- "d" (just a single character)
- "   " (just whitespace)
- "..." (just symbols)
- "问题" (just the word "question" without content)

**EXAMPLES OF VALID CONTENT:**
- Questions: "What are the common systemic symptoms that indicate moderate to severe snake envenomation?"
- Clarification: "Snake bite management requires assessing the severity of envenomation. Systemic symptoms like nausea, tachycardia, and hypotension indicate moderate to severe envenomation, which typically requires antivenom administration."

Follow the Educational Agent Policy. Respond with valid JSON only.
"""
        
        messages = [
            {"role": "system", "content": STUDENT_THINKING_EVALUATION_PROMPT},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM evaluation result
            # LLM will dynamically decide next action (DECOMPOSE or CLARIFY) based on student's current response quality
            result = json.loads(response.choices[0].message.content)
            
            # Print understanding_level from message.content for debugging
            understanding_level = result.get("understanding_level", "unknown")
            print(f"[DEBUG] understanding_level from message.content (evaluate_guidance_response): {understanding_level}")
            
            # ========================================================================
            # Special handling: Detect repeated "I don't know" responses
            # ========================================================================
            # Purpose: If student says "I don't know" multiple times, decomposition may be too complex
            #          Should provide more direct clarification instead of continuing decomposition
            # Strategy: When student says "I don't know" 2 or more times, force CLARIFICATION
            # ========================================================================
            dont_know_keywords = ['毫无头绪', '不知道', '不明白', "i don't know", "don't know", "no idea", 
                                 "not sure", "have no idea", "不清楚", "不懂", "没头绪", "不清楚"]
            
            # Count "I don't know" responses in conversation history
            previous_dont_know_count = sum([1 for m in conversation_history if m.get('role') == 'user' 
                                           and any(keyword in m.get('content', '').lower() for keyword in dont_know_keywords)]) if conversation_history else 0
            # Check if current response is "I don't know"
            current_is_dont_know = any(keyword in student_response.lower() for keyword in dont_know_keywords)
            # Calculate total count (history + current response)
            total_dont_know_count = previous_dont_know_count + (1 if current_is_dont_know else 0)
            
            # If student said "I don't know" 2 or more times, and LLM suggests decompose, force clarify
            # This avoids infinite decomposition and provides more direct help
            if total_dont_know_count >= 2 and result.get("action_type") == "decompose":
                print(f"[DEBUG] Detected {total_dont_know_count} 'I don't know' responses - forcing CLARIFICATION instead of DECOMPOSE")
                # Override action_type to force clarification
                result["action_type"] = "clarify"
                # Generate valid clarification - ensure it's not empty or just symbols
                default_clarification = f"""You've expressed uncertainty multiple times, which is completely normal when learning complex medical concepts. Let me provide a direct explanation with examples:

{result.get('missing_concept', 'The key concept')}

Let me explain this step-by-step with a practical example that relates to the question."""
                clarification = result.get("clarification", "") or default_clarification
                # Validate clarification before using
                if not self._is_clarification_valid(clarification):
                    # If invalid, use missing_concept if available
                    missing_concept = result.get('missing_concept', '')
                    if missing_concept and len(missing_concept.strip()) > 10:
                        clarification = f"{missing_concept}. Understanding these concepts will help you answer the question correctly."
                    else:
                        clarification = ""  # Will trigger flow end
                result["clarification"] = clarification
            
            # Note: Allow decompose even if clarification exists in history - system decides based on response quality
            
            # Determine if student understands
            # Continue decomposition until student understands OR cannot decompose further
            understood = result.get("understanding_level") == "understood" or \
                        result.get("understood", False)
            
            # Check if cannot decompose further
            # Flow termination conditions:
            # 1. Student understands -> TERMINATE
            # 2. Student requests answer -> TERMINATE
            # 
            # Decomposition end conditions:
            # 1. Cannot decompose further -> Use clarification, but flow continues
            # 2. Student understands -> TERMINATE
            
            cannot_decompose = (result.get("action_type") == "decompose" and not result.get("sub_questions")) or \
                              result.get("cannot_decompose_further", False) or \
                              (result.get("action_type") == "decompose" and result.get("reasoning", "").lower().find("cannot") >= 0 and result.get("reasoning", "").lower().find("decompose") >= 0)
            
            # ========================================================================
            # Flow termination condition check: Two main termination conditions
            # ========================================================================
            # Condition 1: Student understands (understood = True)
            #        - Indicates student has grasped key concepts, can end guidance flow
            #        - Return summary of learning content
            #        - Set flow_terminated = True and next_action_type = None to indicate flow end
            # ========================================================================
            if understood:
                # Flow termination condition 1: Student understands - terminate flow
                return {
                    "understood": True,
                    "understanding_level": "understood",
                    "feedback": self._get_validated_feedback(result.get("feedback", "Excellent! You've demonstrated understanding of the key concepts.")),
                    "next_action_type": None,  # None means flow terminates
                    "next_sub_questions": [],
                    "next_clarification": "",
                    "summary": self._get_validated_summary(result.get('missing_concept', 'the main concepts')),
                    "flow_terminated": True  # Explicit flag for flow termination
                }
            else:
                # ========================================================================
                # Student not yet understood - continue guidance flow
                # ========================================================================
                # Important: If cannot decompose further (cannot_decompose_further = True),
                #            it means system has decomposed to most basic level, should try providing clarification
                #            If clarification cannot be provided (invalid content or similar to history), terminate flow
                # ========================================================================
                if cannot_decompose_further or cannot_decompose:
                    # Flow termination condition 2, first part: Cannot decompose further - try providing clarification
                    # Generate clarification, but must meet these conditions:
                    # 1. Content is valid (not empty, not just symbols, not just single characters)
                    # 2. Not similar to clarifications in history (avoid repetition)
                    clarification = self._ensure_valid_clarification(
                        result.get('missing_concept', ''),
                        result.get('feedback', ''),
                        request,
                        conversation_history,
                        current_sub_questions,
                        existing_clarifications  # Pass existing clarifications to check similarity
                    )
                    
                    # Check if clarification is similar to existing ones
                    if clarification and self._check_clarification_similarity(clarification, existing_clarifications):
                        print(f"[DEBUG] Generated clarification is too similar to existing ones - cannot provide new clarification")
                        clarification = ""  # Mark as invalid
                    
                    # Flow termination condition 2 check: Cannot decompose further AND cannot provide clarification
                    # If clarification is invalid (empty or validation failed), terminate flow
                    # Don't reveal answer, just end flow, let student review discussed content
                    if not clarification or not self._is_clarification_valid(clarification):
                        print(f"[DEBUG] Cannot decompose further AND cannot provide clarification - terminating flow")
                        return {
                            "understood": False,
                            "understanding_level": "partial",
                            "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                            "next_action_type": None,  # None means flow terminates
                            "next_sub_questions": [],
                            "next_clarification": "",
                            "summary": "",
                            "reveal_answer": False,  # Don't reveal answer, just end flow
                            "flow_terminated": True,  # TERMINATE: Cannot decompose AND cannot clarify
                            "cannot_decompose_further": True
                        }
                    
                    # Can provide clarification - continue flow, wait for student response
                    # Set cannot_decompose_further = True to indicate decomposition has ended, won't decompose further
                    # flow_terminated = False means flow continues, wait for student response after clarification
                    return {
                        "understood": False,
                        "understanding_level": result.get("understanding_level", "partial"),
                        "feedback": self._get_validated_feedback(result.get("feedback", "")),
                        "next_action_type": "clarify",  # Continue flow, wait for student response
                        "next_sub_questions": [],  # No more decomposition, clear sub-questions list
                        "next_clarification": clarification,
                        "summary": "",
                        "flow_terminated": False,  # Don't terminate, continue guidance loop
                        "cannot_decompose_further": True  # Flag indicates decomposition has ended, won't decompose further
                    }
                
                # ========================================================================
                # Normal flow continues: Dynamically decide next action based on student's current response quality
                # ========================================================================
                # Core logic: System can choose decompose or clarify based on student's current response quality
                #            Even if clarification exists in history, if student's current response is too wrong,
                #            system can still choose decompose to further break down concepts
                #            This is the core of dynamic evaluation: evaluation based on current response, not limited by history
                # ========================================================================
                next_action = result.get("action_type", "decompose")
                
                # Branch 1: LLM decides to clarify
                if next_action == "clarify":
                    # Get clarification from LLM result
                    # Note: Need to validate clarification validity, ensure:
                    # 1. Not empty
                    # 2. Not just template phrases (like "Let me clarify", etc.)
                    # 3. Not just symbols or single characters
                    # 4. Not similar to clarifications in history
                    clarification = result.get("clarification", "")
                    
                    # Validate clarification content validity
                    # _is_clarification_valid checks:
                    # - At least 20 characters long
                    # - At least 5 meaningful words
                    # - Cannot be just symbols or single characters
                    # - Cannot be just template phrases
                    if not self._is_clarification_valid(clarification):
                        # If clarification is invalid, try generating from missing_concept
                        # If missing_concept has enough content (>10 chars), use it to construct clarification
                        missing_concept = result.get("missing_concept", "")
                        if missing_concept and len(missing_concept.strip()) > 10:
                            clarification = f"{missing_concept}. {result.get('feedback', 'Apply this understanding to answer the original question.')}"
                        else:
                            # Flow termination condition 2: Cannot provide clarification
                            # If neither valid clarification nor enough missing_concept to generate, terminate flow
                            print(f"[DEBUG] Cannot generate valid clarification - terminating flow")
                            return {
                                "understood": False,
                                "understanding_level": "partial",
                                "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                                "next_action_type": None,
                                "next_sub_questions": [],
                                "next_clarification": "",
                                "summary": "",
                                "reveal_answer": False,
                                "flow_terminated": True  # TERMINATE: Cannot provide clarification
                            }
                    
                    # Check if clarification is similar to existing ones
                    # 检查clarification是否与现有的相似
                    if clarification and self._check_clarification_similarity(clarification, existing_clarifications):
                        print(f"[DEBUG] Generated clarification is too similar to existing ones - cannot provide new clarification")
                        # FLOW TERMINATION CONDITION 2: Cannot provide new clarification (too similar)
                        # 流程中止条件2：无法提供新的clarification（太相似）
                        return {
                            "understood": False,
                            "understanding_level": "partial",
                            "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                            "next_action_type": None,  # None means flow terminates
                            "next_sub_questions": [],
                            "next_clarification": "",
                            "summary": "",
                            "reveal_answer": False,
                            "flow_terminated": True  # TERMINATE: Cannot provide new clarification
                        }
                    
                    # Critical step: Final validation before returning clarification
                    # Even if passed previous checks, validate again here to ensure returned clarification is valid
                    # This is defensive programming: ensure invalid clarification content is not returned
                    if not clarification or not self._is_clarification_valid(clarification):
                        # Flow termination condition 2: Cannot provide clarification (final validation failed)
                        print(f"[DEBUG] Clarification failed final validation - terminating flow: '{clarification[:50] if clarification else 'empty'}'")
                        return {
                            "understood": False,
                            "understanding_level": "partial",
                            "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                            "next_action_type": None,
                            "next_sub_questions": [],
                            "next_clarification": "",
                            "summary": "",
                            "reveal_answer": False,
                            "flow_terminated": True  # TERMINATE: Cannot provide clarification
                        }
                    
                    # Can provide clarification - continue flow, wait for student response
                    return {
                        "understood": False,
                        "understanding_level": result.get("understanding_level", "close"),
                        "feedback": self._get_validated_feedback(result.get("feedback", "")),
                        "next_action_type": "clarify",  # Continue flow, wait for student response
                        "next_sub_questions": [],
                        "next_clarification": clarification,
                        "summary": "",
                        "flow_terminated": False  # Don't terminate - continue guidance loop
                    }
                else:
                    # ========================================================================
                    # Branch 2: LLM decides to decompose
                    # ========================================================================
                    # First check if student explicitly requests to see answer (special termination case)
                    # ========================================================================
                    answer_request_keywords = ['show me the answer', 'tell me the answer', 'what is the answer', 
                                               'give me the answer', 'reveal the answer', 'want to see the answer',
                                               '想看答案', '告诉我答案', '答案是什么', '给我答案', 'can i see the answer',
                                               'i want the answer', 'please give me the answer']
                    student_wants_answer = any(keyword in student_response.lower() for keyword in answer_request_keywords)
                    
                    # If student explicitly requests answer, terminate flow and reveal answer
                    # This is a special termination condition: respect student's wish
                    if student_wants_answer:
                        return {
                            "understood": True,  # Mark as understood to terminate flow
                            "understanding_level": "requested",
                            "feedback": "I understand you'd like to see the answer. Here it is:",
                            "next_action_type": None,  # None means flow terminates
                            "next_sub_questions": [],
                            "next_clarification": f"The correct answer is {request.correct_answer}: {request.choices.get(request.correct_answer, 'Unknown')}",
                            "summary": "",
                            "reveal_answer": True,  # Flag to indicate answer was revealed
                            "correct_answer": request.correct_answer,
                            "correct_answer_text": request.choices.get(request.correct_answer, 'Unknown'),
                            "flow_terminated": True  # Explicit flag for flow termination
                        }
                    
                    # ========================================================================
                    # Critical check: If cannot decompose further, never try to decompose again
                    # ========================================================================
                    # cannot_decompose_further flag indicates previous decomposition has reached most basic level
                    # Should try providing clarification at this point, not continue trying to decompose
                    # ========================================================================
                    if cannot_decompose_further:
                        clarification = self._ensure_valid_clarification(
                            result.get('missing_concept', ''),
                            result.get('feedback', ''),
                            request,
                            conversation_history,
                            current_sub_questions,
                            existing_clarifications  # Pass existing clarifications to check similarity
                        )
                        
                        # Check if clarification is similar to existing ones
                        if clarification and self._check_clarification_similarity(clarification, existing_clarifications):
                            print(f"[DEBUG] Generated clarification is too similar to existing ones - cannot provide new clarification")
                            clarification = ""  # Mark as invalid
                        
                        # FLOW TERMINATION CONDITION 2: Cannot decompose further AND cannot provide clarification
                        if not clarification or not self._is_clarification_valid(clarification):
                            print(f"[DEBUG] Cannot decompose further AND cannot provide clarification - terminating flow")
                            return {
                                "understood": False,
                                "understanding_level": "partial",
                                "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                                "next_action_type": None,
                                "next_sub_questions": [],
                                "next_clarification": "",
                                "summary": "",
                                "reveal_answer": False,  # Don't reveal answer, just end flow
                                "flow_terminated": True,  # TERMINATE: Cannot decompose AND cannot clarify
                                "cannot_decompose_further": True
                            }
                        
                        # Can provide clarification - continue flow, wait for student response
                        # Critical: After providing clarification, flow should terminate (only one evaluation)
                        return {
                            "understood": False,
                            "understanding_level": result.get("understanding_level", "partial"),
                            "feedback": self._get_validated_feedback(result.get("feedback", "")),
                            "next_action_type": None,  # None means flow terminates after this clarification
                            "next_sub_questions": [],
                            "next_clarification": clarification,
                            "summary": "",
                            "flow_terminated": True,  # CRITICAL: Terminate flow after this clarification (only one evaluation)
                            "cannot_decompose_further": True
                        }
                    
                    # ========================================================================
                    # Normal decomposition flow: Choose different decomposition strategies based on round
                    # ========================================================================
                    # Problem decomposition strategy:
                    # - Round 1: Use generate_sub_questions for MedTutor-R1 style initial decomposition
                    #   This generates complete reasoning step chain, including Observation -> Interpretation -> Conclusion logical flow
                    # - Round 2+: Use dynamic generation method (similar to callSocraticContinuation)
                    #   Based on specific errors and misconceptions in student's current response, generate more targeted sub-questions
                    #   These questions must be simpler and more foundational than previous ones, avoid repetition
                    # ========================================================================
                    if round_number == 1:
                        # First round decomposition: Use MedTutor-R1 method to generate complete reasoning step chain
                        try:
                            decomposition_response = self.generate_sub_questions(request)
                            decomposition = decomposition_response.decomposition
                            
                            # Extract key_questions from reasoning steps
                            sub_questions = [step.key_question for step in decomposition.reasoning_steps]
                            
                            if sub_questions:
                                # CRITICAL: Validate all sub_questions before returning
                                validated_sub_questions = [q for q in sub_questions if self._validate_question(q)]
                                if not validated_sub_questions:
                                    print(f"[DEBUG] All {len(sub_questions)} sub_questions failed validation - ending flow")
                                    return {
                                        "understood": False,
                                        "understanding_level": "partial",
                                        "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                                        "next_action_type": None,
                                        "next_sub_questions": [],
                                        "next_clarification": "",
                                        "summary": "",
                                        "reveal_answer": False,
                                        "flow_terminated": True,
                                        "cannot_decompose_further": True
                                    }
                                
                                # Provide sub-questions and continue flow - wait for student to answer
                                return {
                                    "understood": False,
                                    "understanding_level": result.get("understanding_level", "partial"),
                                    "feedback": self._get_validated_feedback(result.get("feedback", "")),
                                    "next_action_type": "decompose",  # Continue flow, wait for student response
                                    "next_sub_questions": validated_sub_questions,
                                    "next_clarification": "",
                                    "summary": "",
                                    "flow_terminated": False  # Don't terminate - continue guidance loop
                                }
                        except Exception as e:
                            print(f"Error in first-round decomposition: {e}")
                    
                    # Subsequent rounds: Use original recursive decomposition method
                    # This follows the same principles as initial decomposition (MedTutor-R1 style)
                    
                    # Find the most recent question the student struggled with
                    # Use the last question from current_sub_questions as the parent step
                    if current_sub_questions and len(current_sub_questions) > 0:
                        # Use the last question as the step to decompose
                        parent_question = current_sub_questions[-1]
                        
                        # Create a ReasoningStep object for recursive decomposition
                        parent_step = ReasoningStep(
                            step_id=f"{round_number}",
                            key_question=parent_question,
                            step_summary=f"Previous question from round {round_number}",
                            expected_understanding="Student should understand the foundational concept to answer the original question"
                        )
                        
                        # Use the original evaluate_response method for recursive decomposition
                        try:
                            evaluation_result = self.evaluate_response(
                                request=request,
                                step=parent_step,
                                student_response=student_response,
                                previous_questions=current_sub_questions,
                                conversation_history=conversation_history
                            )
                            
                            # Extract simpler steps from the evaluation result
                            if hasattr(evaluation_result, 'sub_steps'):
                                if not evaluation_result.sub_steps or len(evaluation_result.sub_steps) == 0:
                                    # Cannot decompose further - use clarification but flow continues
                                    # FLOW TERMINATION: Only when student understands OR requests answer
                                    return {
                                        "understood": False,
                                        "understanding_level": result.get("understanding_level", "partial"),
                                        "feedback": self._get_validated_feedback(getattr(evaluation_result, 'feedback', result.get("feedback", ""))),
                                        "next_action_type": None,  # None means flow terminates after this clarification
                                        "next_sub_questions": [],
                                        "next_clarification": f"{getattr(evaluation_result, 'missing_concept', result.get('missing_concept', 'the key concepts we have explored'))}. {getattr(evaluation_result, 'feedback', result.get('feedback', 'Apply this understanding to answer the original question.'))}",
                                        "summary": "",
                                        "flow_terminated": True,  # Terminate flow - cannot decompose further
                                        "cannot_decompose_further": True
                                    }
                                
                                simpler_questions = [step.key_question for step in evaluation_result.sub_steps]
                                
                                if simpler_questions:
                                    # CRITICAL: Check if newly generated questions are similar/repetitive
                                    # Collect ALL previously asked questions from ALL rounds
                                    all_previous_questions = []
                                    
                                    # Add ALL questions from current_sub_questions (all rounds, no limit)
                                    if current_sub_questions:
                                        all_previous_questions.extend(current_sub_questions)
                                    
                                    # Extract ALL assistant questions from ENTIRE conversation history (all rounds)
                                    for msg in conversation_history:
                                        if msg.get('role') == 'assistant':
                                            content = msg.get('content', '')
                                            import re
                                            # Extract all questions (not just recent ones)
                                            question_patterns = re.findall(r'[\d+\.]?\s*[🤔❓]\s*(.+?)(?:\n|$)', content)
                                            all_previous_questions.extend([q.strip() for q in question_patterns if q.strip()])
                                    
                                    # Remove duplicates while preserving order
                                    seen = set()
                                    all_previous_questions = [q for q in all_previous_questions if q and q not in seen and not seen.add(q)]
                                    
                                    print(f"[DEBUG] Collected {len(all_previous_questions)} previous questions from ALL rounds for comparison")
                                    
                                    # Check similarity
                                    def questions_are_similar(q1: str, q2: str, threshold: float = 0.3) -> bool:
                                        """
                                        STRICT similarity check - lower threshold (0.3) to catch more repetitions.
                                        严格的相似度检查 - 降低阈值（0.3）以捕获更多重复。
                                        """
                                        if not q1 or not q2:
                                            return False
                                        import string
                                        normalize = lambda s: ''.join(c.lower() for c in s if c not in string.punctuation)
                                        q1_norm = normalize(q1)
                                        q2_norm = normalize(q2)
                                        if q1_norm == q2_norm:
                                            return True
                                        if len(q1_norm) > 10 and len(q2_norm) > 10:
                                            shorter = q1_norm if len(q1_norm) < len(q2_norm) else q2_norm
                                            longer = q2_norm if len(q1_norm) < len(q2_norm) else q1_norm
                                            if len(shorter) >= 15:
                                                if shorter in longer or longer in shorter:
                                                    return True
                                                if len(shorter.split()) >= 4:
                                                    shorter_words = set(shorter.split())
                                                    longer_words = set(longer.split())
                                                    common_words_set = shorter_words.intersection(longer_words)
                                                    if len(common_words_set) >= len(shorter_words) * 0.7:
                                                        return True
                                        # Check generic patterns
                                        generic_patterns = [
                                            ('基本概念', '基础概念', '核心概念', '概念'),
                                            ('应用于临床场景', '应用在临床', '临床应用', '临床应用场景', '临床'),
                                            ('关键特征', '主要特征', '重要特征', '特征'),
                                            ('常见症状', '典型症状', '症状'),
                                            ('严重程度', '严重性', '程度'),
                                            ('治疗方案', '治疗方法', '治疗', '处理'),
                                            ('是什么', '什么是'),
                                            ('为什么', '为何'),
                                            ('如何', '怎样', '怎么'),
                                            ('哪些', '什么'),
                                            ('区别', '不同', '差异')
                                        ]
                                        for pattern_group in generic_patterns:
                                            q1_has = any(p in q1_norm for p in pattern_group)
                                            q2_has = any(p in q2_norm for p in pattern_group)
                                            if q1_has and q2_has:
                                                q1_core = q1_norm
                                                q2_core = q2_norm
                                                for p in pattern_group:
                                                    q1_core = q1_core.replace(p, '')
                                                    q2_core = q2_core.replace(p, '')
                                                if len(q1_core) > 0 and len(q2_core) > 0:
                                                    if len(q1_core) <= 25 and len(q2_core) <= 25:
                                                        return True
                                        words1 = set(q1_norm.split())
                                        words2 = set(q2_norm.split())
                                        common_words = {'what', 'the', 'is', 'are', 'how', 'why', 'when', 'where', 
                                                       'this', 'that', 'these', 'those', 'a', 'an', 'in', 'on', 'at',
                                                       'to', 'for', 'of', 'with', 'from', 'by', 'and', 'or', 'but',
                                                       'can', 'does', 'do', 'will', 'would', 'could', 'should',
                                                       '这里', '这个', '这些', '那些', '什么', '如何', '为什么',
                                                       '是', '的', '了', '和', '与', '或', '但', '在', '从', '到'}
                                        words1 = words1 - common_words
                                        words2 = words2 - common_words
                                        if len(words1) == 0 or len(words2) == 0:
                                            if len(words1) == 0 and len(words2) == 0:
                                                return True
                                            words1 = set(q1_norm.split())
                                            words2 = set(q2_norm.split())
                                        intersection = words1.intersection(words2)
                                        union = words1.union(words2)
                                        if len(union) > 0:
                                            similarity = len(intersection) / len(union)
                                            return similarity >= threshold
                                        # Check medical keywords similarity
                                        medical_keywords1 = {w for w in words1 if len(w) > 4}
                                        medical_keywords2 = {w for w in words2 if len(w) > 4}
                                        if len(medical_keywords1) > 0 and len(medical_keywords2) > 0:
                                            medical_intersection = medical_keywords1.intersection(medical_keywords2)
                                            medical_union = medical_keywords1.union(medical_keywords2)
                                            if len(medical_union) > 0:
                                                medical_similarity = len(medical_intersection) / len(medical_union)
                                                if medical_similarity >= 0.5:
                                                    return True
                                        return False
                                    
                                    # STRICT: Check if questions are repetitive (only check the 2 limited questions)
                                    # THIS CHECK IS MANDATORY - AUTOMATIC REJECTION IF ANY SIMILARITY DETECTED
                                    has_repetitive = False
                                    repetitive_count = 0
                                    repetitive_details = []
                                    
                                    for new_q in simpler_questions:
                                        for prev_q in all_previous_questions:
                                            if questions_are_similar(new_q, prev_q, threshold=0.3):  # STRICT threshold
                                                repetitive_details.append((new_q[:50], prev_q[:50]))
                                                repetitive_count += 1
                                                has_repetitive = True
                                                break  # Stop checking this question once similarity found
                                    
                                    # STRICT: If ANY question is repetitive, stop decomposition immediately
                                    if has_repetitive:
                                        print(f"[WARNING] Repetitive questions detected: {repetitive_count}/{len(simpler_questions)} questions are similar to previous ones. Stopping decomposition.")
                                        return {
                                            "understood": False,
                                            "understanding_level": result.get("understanding_level", "partial"),
                                            "feedback": getattr(evaluation_result, 'feedback', result.get("feedback", "")),
                                            "next_action_type": "clarify",
                                            "next_sub_questions": [],
                                            "next_clarification": f"{getattr(evaluation_result, 'missing_concept', result.get('missing_concept', 'the key concepts we have explored'))}. {getattr(evaluation_result, 'feedback', result.get('feedback', 'Apply this understanding to answer the original question.'))}",
                                            "summary": "",
                                            "cannot_decompose_further": True
                                        }
                                    
                                    # Note: Allow decompose even if clarification exists in history - system decides based on response quality
                                    # 注意：即使历史记录中有clarification，也允许decompose - 系统根据回答质量决定
                                    # CRITICAL: Validate all questions before returning
                                    # 关键：返回前验证所有问题
                                    validated_questions = [q for q in simpler_questions if self._validate_question(q)]
                                    if not validated_questions:
                                        print(f"[DEBUG] All {len(simpler_questions)} questions failed validation - ending flow")
                                        return {
                                            "understood": False,
                                            "understanding_level": "partial",
                                            "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                                            "next_action_type": None,
                                            "next_sub_questions": [],
                                            "next_clarification": "",
                                            "summary": "",
                                            "reveal_answer": False,
                                            "flow_terminated": True,
                                            "cannot_decompose_further": True
                                        }
                                    
                                    # Provide sub-questions and continue flow - wait for student to answer
                                    # 提供子问题并继续流程 - 等待学生回答
                                    return {
                                        "understood": False,
                                        "understanding_level": result.get("understanding_level", "partial"),
                                        "feedback": self._get_validated_feedback(evaluation_result.feedback if hasattr(evaluation_result, 'feedback') else result.get("feedback", "")),
                                        "next_action_type": "decompose",  # Continue flow, wait for student response
                                        "next_sub_questions": validated_questions,
                                        "next_clarification": "",
                                        "summary": "",
                                        "flow_terminated": False  # Don't terminate - continue guidance loop
                                    }
                        except Exception as e:
                            print(f"Error in recursive decomposition using evaluate_response: {e}")
                            # Fall through to alternative method
                    
                    # Alternative: Use RECURSIVE_DECOMPOSITION_PROMPT directly if evaluate_response fails
                    if current_sub_questions and len(current_sub_questions) > 0:
                        parent_question = current_sub_questions[-1]
                        
                        decomp_prompt = RECURSIVE_DECOMPOSITION_PROMPT.format(
                            step_question=parent_question,
                            expected_understanding="Student should understand the foundational concept to answer the original question",
                            student_response=student_response,
                            parent_id=f"{round_number}"
                        )
                        
                        # Collect ALL previously asked questions from ALL sources
                        all_previous_questions_list = []
                        if current_sub_questions:
                            all_previous_questions_list.extend(current_sub_questions)
                        for msg in conversation_history:
                            if msg.get('role') == 'assistant':
                                content = msg.get('content', '')
                                import re
                                question_patterns = re.findall(r'[\d+\.]?\s*[🤔❓]\s*(.+?)(?:\n|$)', content)
                                all_previous_questions_list.extend([q.strip() for q in question_patterns if q.strip()])
                        # Remove duplicates
                        seen_questions = set()
                        all_previous_questions_list = [q for q in all_previous_questions_list if q and q not in seen_questions and not seen_questions.add(q)]
                        
                        decomp_prompt += f"""

## Original Medical Context:
{request.question[:300]}

## Answer Choices:
{chr(10).join([f"{k}: {v}" for k, v in list(request.choices.items())[:4]])}

## COMPLETE Conversation History (CRITICAL - use this for progressive, targeted decomposition):
{chr(10).join([f"Round {i+1} - {m.get('role', 'unknown').upper()}: {m.get('content', '')[:300]}" for i, m in enumerate(conversation_history)]) if conversation_history else "None - this is the first round"}

## ⚠️⚠️⚠️ CRITICAL: ALL PREVIOUSLY ASKED QUESTIONS (MUST CHECK EVERY SINGLE ONE) ⚠️⚠️⚠️
## ⚠️⚠️⚠️ 关键：所有之前问过的问题（必须检查每一个）⚠️⚠️⚠️
## YOU MUST NOT GENERATE ANY QUESTION THAT IS SIMILAR TO ANY OF THESE:
## 你绝对不能生成与以下任何问题相似的问题：
{chr(10).join([f"  [{i+1}] {q}" for i, q in enumerate(all_previous_questions_list)]) if all_previous_questions_list else "  None - this is the first round"}

## Student's Previous Errors and Misconceptions (from conversation history):
{chr(10).join([f"- Round {i+1}: {m.get('content', '')}" for i, m in enumerate(conversation_history) if m.get('role') == 'user']) if conversation_history else "None yet"}

## Current Understanding Level:
{current_understanding_level}

## ⚠️⚠️⚠️ CRITICAL RULES FOR PROGRESSIVE, TARGETED DECOMPOSITION ⚠️⚠️⚠️
## ⚠️⚠️⚠️ 渐进式、针对性分解的关键规则 ⚠️⚠️⚠️

## 🚨🚨🚨 AUTOMATIC REJECTION SYSTEM - YOUR RESPONSE WILL BE CHECKED 🚨🚨🚨
**IMPORTANT: We have AUTOMATIC similarity detection that will REJECT your ENTIRE response if ANY question is similar to previous ones.**
- **STRICT similarity threshold: 0.3** - Very sensitive, will catch even slight similarities
- **AUTOMATIC CHECKING**: Every question you generate will be compared against ALL previous questions
- **IF ANY QUESTION IS SIMILAR**: Your ENTIRE response will be REJECTED, decomposition will STOP, and we will switch to clarification
- **NO EXCEPTIONS**: Even if you think a question is "different enough", if similarity >= 0.3, it will be REJECTED
- **THE ONLY WAY TO SUCCEED**: Generate COMPLETELY NEW questions that explore DIFFERENT concepts or angles

1. **ABSOLUTE PROHIBITION - NO SIMILAR QUESTIONS**:
   - You MUST check EVERY question in the "ALL PREVIOUSLY ASKED QUESTIONS" list above
   - Your new questions MUST be GENUINELY DIFFERENT from ALL previous questions
   - DO NOT generate questions that are rephrased versions, similar concepts, or generic patterns
   - **YOUR RESPONSE WILL BE AUTOMATICALLY REJECTED** if similarity detection finds ANY similarity (threshold 0.3)
   - If you cannot generate genuinely NEW and DIFFERENT questions, you MUST set "cannot_decompose_further": true

2. **FORBIDDEN QUESTION PATTERNS** (ABSOLUTELY DO NOT GENERATE - THESE WILL BE REJECTED):
   - "这里涉及的基本概念是什么" / "What is the basic concept here"
   - "这个概念如何应用于临床场景" / "How does this concept apply to clinical scenarios"
   - "让我把它分解成更简单的步骤" / "Let me break it down into simpler steps"
   - "这里涉及的基本概念" / "基本概念是什么" / "如何应用于临床场景"
   - Any generic question that doesn't use specific medical terminology from the original question
   - Any question that is similar (even slightly) to any question in the list above
   - **IF YOU GENERATE ANY OF THESE, YOUR RESPONSE WILL BE REJECTED AND DECOMPOSITION WILL STOP**

3. **MANDATORY REQUIREMENTS**:
   - **MUST use conversation history**: Your new questions must build on what has been discussed
   - **Target student's specific errors**: Address the misconceptions shown in their previous responses
   - Sub-steps must be SIMPLER and more FOUNDATIONAL than "{parent_question}"
   - Must explore DIFFERENT concepts (not rephrased versions) - check ALL previous questions above
   - Follow the same principles as initial decomposition (Holistic Analysis, Chain of Reasoning, etc.)
   - Generate at most 2 simpler steps per round (maximum 2 to avoid overwhelming the student, but ensure each step is genuinely different from ALL previous questions)
   - **Strengthen correction**: Focus on correcting the specific errors the student has made
   - **Each round must be more targeted**: Based on what you learned from previous rounds about the student's gaps

4. **IF YOU CANNOT DECOMPOSE FURTHER**:
   - If you cannot generate genuinely NEW and DIFFERENT questions that explore different concepts
   - If all questions you can think of are similar to previous ones
   - If the problem cannot be broken down further
   - **YOU MUST**: Set "cannot_decompose_further": true in your response
   - **YOU MUST**: Provide meaningful "feedback" and "missing_concept" (NOT blank, NOT empty, NOT generic)
   - **DO NOT**: Return empty simpler_steps, generic questions, or blank content
   - **DO NOT**: Return questions that are similar to any in the list above
   - **DO NOT**: Return generic questions like "基本概念是什么" or "如何应用于临床场景" - these will be automatically rejected

5. **RESPONSE REQUIREMENTS**:
   - If "cannot_decompose_further": true, you may return empty simpler_steps: []
   - But you MUST provide meaningful "feedback" and "missing_concept" (NOT blank)
   - NEVER return empty or generic content
"""
                        
                        try:
                            decomp_response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[
                                    {"role": "system", "content": RECURSIVE_DECOMPOSITION_PROMPT.split("## Response Format")[0] + "\n\n⚠️⚠️⚠️ CRITICAL LIMIT ⚠️⚠️⚠️: You MUST generate EXACTLY 1-2 simpler sub-steps (MAXIMUM 2). Do NOT generate more than 2. If you generate more than 2, your response will be REJECTED.\n\n## Response Format:\nYou must respond with valid JSON only. The JSON must contain a 'simpler_steps' array with EXACTLY 1-2 items (MAXIMUM 2)."},
                                    {"role": "user", "content": decomp_prompt}
                                ],
                                temperature=0.5,
                                max_tokens=2000,  # Increased to support generating many sub-steps without limit
                                response_format={"type": "json_object"}
                            )
                            
                            decomp_result = json.loads(decomp_response.choices[0].message.content)
                            simpler_steps = decomp_result.get("simpler_steps", [])
                            cannot_decompose = decomp_result.get("cannot_decompose_further", False)
                            
                            # CRITICAL: Check if newly generated questions are similar/repetitive to previously asked questions
                            # 关键：检查新生成的问题是否与之前问过的问题相似/重复
                            if simpler_steps:
                                # Collect all previously asked questions from conversation history and current_sub_questions
                                # 从对话历史和current_sub_questions中收集所有之前问过的问题
                                all_previous_questions = []
                                
                                # Add questions from current_sub_questions
                                if current_sub_questions:
                                    all_previous_questions.extend(current_sub_questions)
                                
                                # Extract assistant questions from conversation history
                                # 从对话历史中提取助手的问题
                                for msg in conversation_history:
                                    if msg.get('role') == 'assistant':
                                        content = msg.get('content', '')
                                        # Try to extract questions from assistant messages
                                        # 尝试从助手消息中提取问题
                                        # Look for patterns like "1. 🤔 [question]" or "🤔 [question]"
                                        import re
                                        question_patterns = re.findall(r'[\d+\.]?\s*[🤔❓]\s*(.+?)(?:\n|$)', content)
                                        all_previous_questions.extend([q.strip() for q in question_patterns if q.strip()])
                                
                                # CRITICAL: Check similarity of new questions to previous ones
                                # 关键：检查新问题与之前问题的相似度
                                # This check is MANDATORY - if ANY question is similar, ENTIRE response is REJECTED
                                # 此检查是强制性的 - 如果任何问题相似，整个响应将被拒绝
                                new_questions = [step.get("key_question", "").strip() for step in simpler_steps if step.get("key_question")]
                                
                                def questions_are_similar(q1: str, q2: str, threshold: float = 0.3) -> bool:
                                    """
                                    STRICT similarity check - lower threshold (0.3) to catch more repetitions.
                                    严格的相似度检查 - 降低阈值（0.3）以捕获更多重复。
                                    Uses multiple methods: exact match, substring, word overlap, and semantic keywords.
                                    使用多种方法：精确匹配、子串、词汇重叠和语义关键词。
                                    """
                                    if not q1 or not q2:
                                        return False
                                    
                                    # Normalize: lowercase, remove punctuation
                                    # 标准化：小写，去除标点
                                    import string
                                    normalize = lambda s: ''.join(c.lower() for c in s if c not in string.punctuation)
                                    q1_norm = normalize(q1)
                                    q2_norm = normalize(q2)
                                    
                                    # 1. Exact match (after normalization)
                                    # 1. 精确匹配（标准化后）
                                    if q1_norm == q2_norm:
                                        return True
                                    
                                    # 2. Check if one is a substring of the other (very similar)
                                    # 2. 检查一个是否是另一个的子串（非常相似）
                                    if len(q1_norm) > 10 and len(q2_norm) > 10:
                                        # Check if 70% or more of shorter question is contained in longer one
                                        # 检查较短问题的70%或更多是否包含在较长问题中
                                        shorter = q1_norm if len(q1_norm) < len(q2_norm) else q2_norm
                                        longer = q2_norm if len(q1_norm) < len(q2_norm) else q1_norm
                                        if len(shorter) >= 15:
                                            if shorter in longer or longer in shorter:
                                                return True
                                            # Check if key words match (for generic questions)
                                            # 检查关键词是否匹配（针对通用问题）
                                            if len(shorter.split()) >= 4:
                                                shorter_words = set(shorter.split())
                                                longer_words = set(longer.split())
                                                common_words_set = shorter_words.intersection(longer_words)
                                                if len(common_words_set) >= len(shorter_words) * 0.7:
                                                    return True
                                    
                                    # 3. Check for generic question patterns that are always similar
                                    # 3. 检查始终相似的通用问题模式
                                    generic_patterns = [
                                        ('基本概念', '基础概念', '核心概念', '概念'),
                                        ('应用于临床场景', '应用在临床', '临床应用', '临床应用场景', '临床'),
                                        ('关键特征', '主要特征', '重要特征', '特征'),
                                        ('常见症状', '典型症状', '症状'),
                                        ('严重程度', '严重性', '程度'),
                                        ('治疗方案', '治疗方法', '治疗', '处理'),
                                        ('是什么', '什么是'),
                                        ('为什么', '为何'),
                                        ('如何', '怎样', '怎么'),
                                        ('哪些', '什么'),
                                        ('区别', '不同', '差异')
                                    ]
                                    
                                    for pattern_group in generic_patterns:
                                        q1_has = any(p in q1_norm for p in pattern_group)
                                        q2_has = any(p in q2_norm for p in pattern_group)
                                        if q1_has and q2_has:
                                            # Both have similar generic patterns, check if core structure is similar
                                            # 两者都有相似的通用模式，检查核心结构是否相似
                                            q1_core = q1_norm
                                            q2_core = q2_norm
                                            for p in pattern_group:
                                                q1_core = q1_core.replace(p, '')
                                                q2_core = q2_core.replace(p, '')
                                            if len(q1_core) > 0 and len(q2_core) > 0:
                                                if len(q1_core) <= 25 and len(q2_core) <= 25:
                                                    # Very short remaining text suggests generic repetitive questions
                                                    # 非常短的剩余文本表明是通用重复问题
                                                    return True
                                    
                                    # 4. Word overlap similarity (STRICT - lower threshold)
                                    # 4. 词汇重叠相似度（严格 - 降低阈值）
                                    words1 = set(q1_norm.split())
                                    words2 = set(q2_norm.split())
                                    
                                    if len(words1) == 0 or len(words2) == 0:
                                        return False
                                    
                                    # Remove very common words (stopwords-like)
                                    # 移除非常常见的词（类似停用词）
                                    common_words = {'what', 'the', 'is', 'are', 'how', 'why', 'when', 'where', 
                                                   'this', 'that', 'these', 'those', 'a', 'an', 'in', 'on', 'at',
                                                   'to', 'for', 'of', 'with', 'from', 'by', 'and', 'or', 'but',
                                                   'can', 'does', 'do', 'will', 'would', 'could', 'should',
                                                   '这里', '这个', '这些', '那些', '什么', '如何', '为什么',
                                                   '是', '的', '了', '和', '与', '或', '但', '在', '从', '到'}
                                    words1 = words1 - common_words
                                    words2 = words2 - common_words
                                    
                                    if len(words1) == 0 or len(words2) == 0:
                                        # If after removing common words, both are empty, likely too generic
                                        # 如果移除常见词后，两者都为空，可能太通用
                                        if len(words1) == 0 and len(words2) == 0:
                                            return True
                                        # Use original words if filtered too much
                                        words1 = set(q1_norm.split())
                                        words2 = set(q2_norm.split())
                                    
                                    intersection = words1.intersection(words2)
                                    union = words1.union(words2)
                                    
                                    # Jaccard similarity with STRICT threshold (0.3)
                                    # Jaccard相似度，严格阈值（0.3）
                                    if len(union) > 0:
                                        similarity = len(intersection) / len(union)
                                        # Lower threshold (0.3) to catch more repetitions
                                        # 降低阈值（0.3）以捕获更多重复
                                        return similarity >= threshold
                                    
                                    # 5. Check if questions share too many key medical terms (semantic similarity)
                                    # 5. 检查问题是否共享太多关键医学术语（语义相似度）
                                    # Extract medical/clinical keywords (longer words, likely medical terms)
                                    # 提取医学/临床关键词（较长的词，可能是医学术语）
                                    medical_keywords1 = {w for w in words1 if len(w) > 4}
                                    medical_keywords2 = {w for w in words2 if len(w) > 4}
                                    
                                    if len(medical_keywords1) > 0 and len(medical_keywords2) > 0:
                                        medical_intersection = medical_keywords1.intersection(medical_keywords2)
                                        medical_union = medical_keywords1.union(medical_keywords2)
                                        if len(medical_union) > 0:
                                            medical_similarity = len(medical_intersection) / len(medical_union)
                                            # If medical terms are 50%+ similar, likely repetitive
                                            # 如果医学术语50%+相似，可能是重复的
                                            if medical_similarity >= 0.5:
                                                return True
                                    
                                    return False
                                
                                # STRICT: First validate questions (filter out generic/invalid ones)
                                # 严格：首先验证问题（过滤掉通用/无效的问题）
                                def is_question_valid_strict(q: str) -> bool:
                                    """Strict validation to reject generic questions."""
                                    if not q:
                                        return False
                                    q_clean = q.strip()
                                    if len(q_clean) < 5:
                                        return False
                                    q_lower = q_clean.lower()
                                    # Reject generic question patterns
                                    generic_patterns = [
                                        "这里涉及的基本概念", "这个概念如何应用于临床场景",
                                        "这里涉及的基本概念是什么", "这个概念如何应用于临床",
                                        "基本概念是什么", "如何应用于临床场景",
                                        "让我把它分解成更简单的步骤", "把它分解成更简单的步骤",
                                        "分解成更简单的步骤", "涉及的基本概念", "应用于临床场景"
                                    ]
                                    for pattern in generic_patterns:
                                        if pattern in q_lower or pattern in q_clean:
                                            print(f"[DEBUG] Rejected generic question: '{q_clean[:60]}...'")
                                            return False
                                    return True
                                
                                # Filter out generic/invalid questions BEFORE similarity check
                                # 在相似度检查之前过滤掉通用/无效的问题
                                valid_new_questions = [q for q in new_questions if is_question_valid_strict(q)]
                                
                                # CRITICAL: If no valid questions after filtering, stop immediately
                                # 关键：如果过滤后没有有效问题，立即停止
                                if not valid_new_questions:
                                    print(f"[DEBUG] All {len(new_questions)} generated questions are generic/invalid - stopping decomposition")
                                    cannot_decompose = True
                                else:
                                    # Use only valid questions for similarity checking
                                    new_questions = valid_new_questions
                                
                                # STRICT: Check each new question against all previous questions
                                # 严格：将每个新问题与所有之前的问题进行比较
                                # THIS CHECK IS MANDATORY - AUTOMATIC REJECTION IF ANY SIMILARITY DETECTED
                                # 此检查是强制性的 - 如果检测到任何相似性，将自动拒绝
                                has_repetitive_questions = False
                                repetitive_count = 0
                                
                                for new_q in new_questions:
                                    for prev_q in all_previous_questions:
                                        if questions_are_similar(new_q, prev_q, threshold=0.3):  # STRICT threshold
                                            has_repetitive_questions = True
                                            repetitive_count += 1
                                            break
                                
                                # STRICT: If ANY question is repetitive, stop decomposition immediately
                                # 严格：如果任何问题是重复的，立即停止分解
                                if has_repetitive_questions:
                                    print(f"[WARNING] Repetitive questions detected: {repetitive_count}/{len(new_questions)} questions are similar to previous ones. Stopping decomposition.")
                                    cannot_decompose = True
                            
                            # Check if cannot decompose further (empty steps or flag set or repetitive questions)
                            # 检查是否无法继续分解（空步骤或标志已设置或问题重复）
                            if cannot_decompose or not simpler_steps:
                                # DECOMPOSITION END CONDITION: Cannot decompose further
                                # 解析结束条件：无法继续分解
                                # FLOW TERMINATION: Only when student understands OR requests answer
                                # 流程中止：只有当学生理解或要求答案时
                                # When cannot decompose further, ALWAYS use clarification (flow continues)
                                # 当无法继续分解时，总是使用澄清（流程继续）
                                
                                # CRITICAL: If clarification already exists in history, reveal answer directly
                                # 关键：如果历史记录中已经有clarification，直接给出答案
                                if has_clarification_in_history:
                                    print(f"[DEBUG] Clarification already exists in history - ending flow without revealing answer")
                                    return {
                                        "understood": False,
                                        "understanding_level": "partial",
                                        "feedback": "",
                                        "next_action_type": None,  # None means flow terminates
                                        "next_sub_questions": [],
                                        "next_clarification": "",
                                        "summary": "",
                                        "reveal_answer": False,
                                        "flow_terminated": True,
                                        "cannot_decompose_further": True
                                    }
                                
                                # Generate meaningful clarification using LLM if missing_concept is empty
                                # 如果missing_concept为空，使用LLM生成有意义的clarification
                                missing_concept = decomp_result.get('missing_concept') or result.get('missing_concept', '')
                                feedback_text = decomp_result.get('feedback') or result.get('feedback', '')
                                
                                # If clarification content is empty or incomplete, generate it
                                # 如果clarification内容为空或不完整，生成它
                                if not missing_concept or len(missing_concept.strip()) < 10:
                                    try:
                                        clarification_prompt = f"""You are a medical education tutor. The student has been working through this problem but needs clarification.

## Original Question:
{request.question[:500]}

## Answer Choices:
{chr(10).join([f"{k}: {v}" for k, v in list(request.choices.items())[:4]])}

## Conversation Summary:
{chr(10).join([f"- {m.get('role', 'unknown').upper()}: {m.get('content', '')[:200]}" for m in conversation_history[-4:]]) if conversation_history else "None"}

## Student's Current Understanding Level:
{current_understanding_level}

## Previously Asked Questions:
{chr(10).join([f"- {q}" for q in current_sub_questions[-3:]]) if current_sub_questions else "None"}

Generate a clear, helpful clarification that:
1. Summarizes the key concepts the student needs to understand
2. Provides a direct explanation WITHOUT revealing the correct answer
3. Uses examples or analogies if helpful
4. Encourages the student to apply this understanding

Respond with JSON:
{{
    "clarification": "Your clear, helpful explanation here (2-4 sentences, NEVER reveal the answer)"
}}"""
                                        
                                        clarification_response = self.client.chat.completions.create(
                                            model=self.model,
                                            messages=[
                                                {"role": "system", "content": "You are a medical education tutor. Provide helpful clarifications WITHOUT revealing the correct answer.\n\nCRITICAL VALIDATION RULES (VIOLATION = AUTOMATIC REJECTION):\n- Clarification MUST be at least 20 characters long\n- MUST contain at least 5 meaningful words (words longer than 1 character)\n- NO symbols only (\".\", \"?\", \"!\", etc.)\n- NO single characters (\"h\", \"d\", etc.)\n- NO blanks or whitespace only\n- Must contain actual medical/educational content\n- Invalid clarifications will be automatically rejected and flow will end"},
                                                {"role": "user", "content": clarification_prompt}
                                            ],
                                            temperature=0.7,
                                            max_tokens=300,
                                            response_format={"type": "json_object"}
                                        )
                                        
                                        clarification_result = json.loads(clarification_response.choices[0].message.content)
                                        generated_clarification = clarification_result.get("clarification", "")
                                        
                                        if generated_clarification and len(generated_clarification.strip()) > 20:
                                            final_clarification = generated_clarification
                                        else:
                                            # If clarification is empty, return empty string so system will reveal answer
                                            # 如果clarification为空，返回空字符串，系统将给出答案
                                            final_clarification = ""
                                    except Exception as e:
                                        print(f"Error generating clarification: {e}")
                                        # If error, return empty string so system will reveal answer
                                        # 如果出错，返回空字符串，系统将给出答案
                                        final_clarification = ""
                                else:
                                    # Use existing missing_concept and feedback, but ensure it's not empty
                                    # 使用现有的missing_concept和feedback，但确保不为空
                                    if missing_concept and len(missing_concept.strip()) > 10:
                                        final_clarification = f"{missing_concept}. {feedback_text if feedback_text and len(feedback_text.strip()) > 5 else 'Apply this understanding to answer the original question.'}"
                                    else:
                                        # If missing_concept is still empty, generate it
                                        # 如果missing_concept仍然为空，生成它
                                        final_clarification = self._ensure_valid_clarification(
                                            '', feedback_text, request, conversation_history, current_sub_questions,
                                            existing_clarifications  # Pass existing clarifications to check similarity
                                        )
                                        
                                        # Check if clarification is similar to existing ones
                                        # 检查clarification是否与现有的相似
                                        if final_clarification and self._check_clarification_similarity(final_clarification, existing_clarifications):
                                            print(f"[DEBUG] Generated clarification is too similar to existing ones - ending flow without revealing answer")
                                            final_clarification = ""  # Mark as invalid to end flow
                                
                                # If clarification is empty or invalid, end flow without revealing answer
                                # 如果clarification为空或无效，结束流程但不给出答案
                                if not final_clarification or not self._is_clarification_valid(final_clarification):
                                    print(f"[DEBUG] No valid clarification can be generated - ending flow without revealing answer")
                                    return {
                                        "understood": False,
                                        "understanding_level": "partial",
                                        "feedback": "",
                                        "next_action_type": None,  # None means flow terminates
                                        "next_sub_questions": [],
                                        "next_clarification": "",
                                        "summary": "",
                                        "reveal_answer": False,
                                        "flow_terminated": True,
                                        "cannot_decompose_further": True
                                    }
                                
                                return {
                                    "understood": False,
                                    "understanding_level": result.get("understanding_level", "partial"),
                                    "feedback": self._get_validated_feedback(result.get("feedback", "")),
                                    "next_action_type": None,  # None means flow terminates after this clarification
                                    "next_sub_questions": [],  # NEVER decompose again
                                    "next_clarification": final_clarification,
                                    "summary": "",
                                    "flow_terminated": True,  # Terminate flow - cannot decompose further
                                    "cannot_decompose_further": True  # Flag indicating decomposition ended - NEVER decompose again
                                }
                            
                            if simpler_steps:
                                # Helper function to validate questions (defined here for this scope)
                                # 验证问题的辅助函数（在此作用域内定义）
                                def is_question_valid_local(q: str) -> bool:
                                    """Check if a question is valid (not empty, not too short, has meaningful content, not just symbols)."""
                                    if not q:
                                        return False
                                    q_clean = q.strip()
                                    
                                    # CRITICAL: Check for empty or too short
                                    # 关键：检查是否为空或太短
                                    if len(q_clean) < 5:
                                        return False
                                    
                                    # CRITICAL: Check if question is only punctuation/symbols/whitespace
                                    # 关键：检查问题是否只是标点/符号/空白
                                    import string
                                    if all(c in string.punctuation + string.whitespace for c in q_clean):
                                        print(f"[DEBUG] Rejected question - only punctuation/symbols: '{q_clean}'")
                                        return False
                                    
                                    # CRITICAL: Check if question is just a single character or dot
                                    # 关键：检查问题是否只是单个字符或点
                                    meaningful_chars = q_clean.replace('.', '').replace(' ', '').replace(',', '').replace('!', '').replace('?', '').replace('。', '').replace('，', '').replace('！', '').replace('？', '')
                                    if len(meaningful_chars) < 3:
                                        print(f"[DEBUG] Rejected question - mostly punctuation: '{q_clean}'")
                                        return False
                                    
                                    # CRITICAL: Check if question is just single characters (like "h", "d", etc.)
                                    # 关键：检查问题是否只是单个字符（如"h"、"d"等）
                                    words = [w for w in q_clean.split() if len(w.strip()) > 0]
                                    if len(words) == 0:
                                        return False
                                    # If all words are single characters, reject
                                    if all(len(w.strip()) <= 1 for w in words):
                                        print(f"[DEBUG] Rejected question - only single characters: '{q_clean}'")
                                        return False
                                    
                                    invalid_patterns = ["...", "N/A", "n/a", "TBD", "待定", "待填写",
                                                       "question here", "问题", "问题1", "问题2",
                                                       "key question", "sub-question", "子问题"]
                                    q_lower = q_clean.lower()
                                    if any(pattern in q_lower for pattern in invalid_patterns):
                                        return False
                                    
                                    # CRITICAL: Detect generic questions that should be rejected
                                    # 关键：检测应该被拒绝的通用问题
                                    generic_question_patterns = [
                                        "这里涉及的基本概念",
                                        "这个概念如何应用于临床场景",
                                        "这里涉及的基本概念是什么",
                                        "这个概念如何应用于临床",
                                        "基本概念是什么",
                                        "如何应用于临床场景",
                                        "让我把它分解成更简单的步骤",
                                        "把它分解成更简单的步骤",
                                        "分解成更简单的步骤",
                                        "涉及的基本概念",
                                        "应用于临床场景"
                                    ]
                                    for pattern in generic_question_patterns:
                                        if pattern in q_lower or pattern in q_clean:
                                            print(f"[DEBUG] Rejected generic question: '{q_clean[:60]}...' (contains pattern: '{pattern}')")
                                            return False
                                    import re
                                    if re.match(r'^[\d\s\.\?\!\,\;\:\-]+$', q_clean):
                                        return False
                                    return True
                                
                                # Extract and validate questions - filter out empty/invalid ones
                                # 提取并验证问题 - 过滤掉空/无效的问题
                                raw_questions = [step.get("key_question", "") for step in simpler_steps if step.get("key_question")]
                                
                                # Filter out invalid questions (empty, too short, placeholders)
                                # 过滤掉无效问题（空、太短、占位符）
                                simpler_questions = [q for q in raw_questions if is_question_valid_local(q)]
                                
                                # CRITICAL: If no valid questions generated, stop decomposition immediately and end flow
                                # 关键：如果没有生成有效问题，立即停止分解并结束流程
                                if not simpler_questions:
                                    print(f"[DEBUG] No valid questions generated from {len(raw_questions)} raw questions - ending flow")
                                    return {
                                        "understood": False,
                                        "understanding_level": "partial",
                                        "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                                        "next_action_type": None,
                                        "next_sub_questions": [],
                                        "next_clarification": "",
                                        "summary": "",
                                        "reveal_answer": False,
                                        "flow_terminated": True,
                                        "cannot_decompose_further": True
                                    }
                                
                                if simpler_questions:
                                    # Note: Allow decompose even if clarification exists in history - system decides based on response quality
                                    # 注意：即使历史记录中有clarification，也允许decompose - 系统根据回答质量决定
                                    # CRITICAL: Validate all questions before returning
                                    # 关键：返回前验证所有问题
                                    validated_questions = [q for q in simpler_questions if self._validate_question(q)]
                                    if not validated_questions:
                                        print(f"[DEBUG] All {len(simpler_questions)} questions failed validation - ending flow")
                                        return {
                                            "understood": False,
                                            "understanding_level": "partial",
                                            "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                                            "next_action_type": None,
                                            "next_sub_questions": [],
                                            "next_clarification": "",
                                            "summary": "",
                                            "reveal_answer": False,
                                            "flow_terminated": True,
                                            "cannot_decompose_further": True
                                    }
                                
                                    # Provide sub-questions and continue flow - wait for student to answer
                                    # 提供子问题并继续流程 - 等待学生回答
                                    return {
                                        "understood": False,
                                        "understanding_level": result.get("understanding_level", "partial"),
                                        "feedback": self._get_validated_feedback(decomp_result.get("feedback", result.get("feedback", ""))),
                                        "next_action_type": "decompose",  # Continue flow, wait for student response
                                        "next_sub_questions": validated_questions,
                                        "next_clarification": "",
                                        "summary": "",
                                        "flow_terminated": False  # Don't terminate - continue guidance loop
                                    }
                        except Exception as e:
                            print(f"Error in recursive decomposition using RECURSIVE_DECOMPOSITION_PROMPT: {e}")
                    
                    # Final fallback: Use simple questions
                    # 最终回退：使用简单问题
                    # Note: Allow decompose even if clarification exists in history - system decides based on response quality
                    # 注意：即使历史记录中有clarification，也允许decompose - 系统根据回答质量决定
                    # CRITICAL: Validate questions before returning
                    # 关键：返回前验证问题
                    fallback_questions = result.get("sub_questions", ["What are the key concepts involved?"])
                    validated_questions = [q for q in fallback_questions if self._validate_question(q)]
                    if not validated_questions:
                        print(f"[DEBUG] All fallback questions failed validation - ending flow")
                        return {
                            "understood": False,
                            "understanding_level": "partial",
                            "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                            "next_action_type": None,
                            "next_sub_questions": [],
                            "next_clarification": "",
                            "summary": "",
                            "reveal_answer": False,
                            "flow_terminated": True,
                            "cannot_decompose_further": True
                        }
                    
                    # Provide sub-questions and continue flow - wait for student to answer
                    # 提供子问题并继续流程 - 等待学生回答
                    return {
                        "understood": False,
                        "understanding_level": result.get("understanding_level", "partial"),
                        "feedback": self._get_validated_feedback(result.get("feedback", "")),
                        "next_action_type": "decompose",  # Continue flow, wait for student response
                        "next_sub_questions": validated_questions,
                        "next_clarification": "",
                        "summary": "",
                        "flow_terminated": False  # Don't terminate - continue guidance loop
                    }
            
        except Exception as e:
            print(f"Error in evaluate_guidance_response: {e}")
            # Fallback: Try to continue with decomposition or clarification based on error
            # 回退：尝试根据错误继续decompose或clarify
            # Note: Allow decompose even if clarification exists in history - system decides based on response quality
            # 注意：即使历史记录中有clarification，也允许decompose - 系统根据回答质量决定
            # CRITICAL: Fallback - terminate flow only if cannot decompose AND cannot clarify
            # 关键：回退 - 终止流程（只执行一次评估）
            return {
                "understood": False,
                "understanding_level": current_understanding_level,
                "feedback": "I've provided clarification on this topic. Please review the concepts we've discussed and try answering the question again.",
                "next_action_type": None,  # None means flow terminates
                "next_sub_questions": [],
                "next_clarification": "",
                "summary": "",
                "flow_terminated": True  # CRITICAL: Terminate flow (only one evaluation)
            }
    
    def should_trigger_hints(
        self, 
        student_answer: str, 
        correct_answer: str, 
        user_message: Optional[str] = None
    ) -> Tuple[bool, str]:
        """判断是否应该触发提示
        
        触发条件：
        1. 学生答案错误
        2. 学生明确请求提示（通过user_message中的关键词）
        
        参数:
            student_answer: 学生选择的答案
            correct_answer: 正确答案
            user_message: 可选的学生消息（例如："给我提示"、"我不懂"等）
            
        返回:
            元组（是否触发：布尔值，原因：字符串）
            原因可能是："wrong_answer"（答案错误）、"student_request"（学生请求）、"no_trigger"（不触发）
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
        student_response: str,
        previous_questions: List[str] = None,
        conversation_history: List[Dict] = None
    ) -> EvaluationResult:
        """评估学生对推理步骤的回答
        
        如果学生没有表现出理解，则分解为更简单的子步骤。
        
        MedTutor-R1方法：观察 → 解释 → 结论
        
        这个方法用于评估学生对单个推理步骤的回答。如果学生理解了，
        返回成功结果；如果不理解，使用递归分解方法生成更简单的子步骤。
        
        参数:
            request: HintRequest对象
            step: 要评估的推理步骤
            student_response: 学生的回答
            previous_questions: 之前问过的问题列表（用于避免重复）
            conversation_history: 对话历史
            
        返回:
            EvaluationResult对象，包含评估结果和可能的子步骤
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
    "feedback": "Encouraging, specific feedback (NOT generic templates like 'Let me break this down' or 'Does this help?')",
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
            
            # Collect all previously asked questions
            # 收集所有之前问过的问题
            all_previous_questions_list = []
            if previous_questions:
                all_previous_questions_list.extend(previous_questions)
            if conversation_history:
                import re
                for msg in conversation_history:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        question_patterns = re.findall(r'[\d+\.]?\s*[🤔❓]\s*(.+?)(?:\n|$)', content)
                        all_previous_questions_list.extend([q.strip() for q in question_patterns if q.strip()])
            # Remove duplicates
            seen_questions = set()
            all_previous_questions_list = [q for q in all_previous_questions_list if q and q not in seen_questions and not seen_questions.add(q)]
            
            decomp_prompt += f"""

## Original Medical Context:
{request.question[:300]}

## What the student is missing:
{missing}

## ⚠️⚠️⚠️ CRITICAL: ALL PREVIOUSLY ASKED QUESTIONS (MUST CHECK EVERY SINGLE ONE) ⚠️⚠️⚠️
## ⚠️⚠️⚠️ 关键：所有之前问过的问题（必须检查每一个）⚠️⚠️⚠️
## YOU MUST NOT REPEAT ANY OF THESE QUESTIONS - NOT EVEN IF REPHRASED:
{chr(10).join([f"  [{i+1}] {q}" for i, q in enumerate(all_previous_questions_list)]) if all_previous_questions_list else "  None - this is the first round"}

## ⚠️⚠️⚠️ CRITICAL REQUIREMENTS ⚠️⚠️⚠️
1. **MAXIMUM 2 SUB-STEPS PER ROUND**: Generate at most 2 sub-steps per round to avoid overwhelming the student. There is a strict limit of 2 questions per round.
2. **ABSOLUTE PROHIBITION - NO REPETITION**: You MUST NOT repeat ANY question from the list above, even if rephrased. Every single sub-question must be completely new.
3. **EACH SUB-QUESTION MUST BE UNIQUE**: If you cannot generate a genuinely new question that explores a different concept or angle, set "cannot_decompose_further": true.
4. **CRITICAL LIMIT**: Generate EXACTLY 1-2 simpler sub-steps per round (MAXIMUM 2). This is a strict limit. Do NOT generate more than 2 sub-steps.
"""
            
            decomp_messages = [
                {"role": "system", "content": RECURSIVE_DECOMPOSITION_PROMPT.split("## Response Format")[0] + "\n\n⚠️⚠️⚠️ CRITICAL LIMIT ⚠️⚠️⚠️: You MUST generate EXACTLY 1-2 simpler sub-steps (MAXIMUM 2). Do NOT generate more than 2. If you generate more than 2, your response will be REJECTED.\n\n## Response Format:\nYou must respond with valid JSON only. The JSON must contain a 'simpler_steps' array with EXACTLY 1-2 items (MAXIMUM 2)."},
                {"role": "user", "content": decomp_prompt}
            ]
            
            decomp_response = self.client.chat.completions.create(
                model=self.model,
                messages=decomp_messages,
                temperature=0.7,
                max_tokens=2000,  # Sufficient for generating up to 2 sub-steps per round
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
            print(f"Error in evaluate_response: {e}")
            return EvaluationResult(
                step_id=step.step_id,
                understood=False,
                feedback="",
                sub_steps=[],
                missing_concept=""
            )
    
    def format_hints_for_display(self, response: SocraticResponse) -> str:
        """格式化推理步骤以便在UI中显示（MedTutor-R1风格）
        
        参数:
            response: SocraticResponse对象
            
        返回:
            格式化后的显示文本
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
                status = " 🔄"  # 需要更多努力
            else:
                status = " ⏳"  # 尚未回答
            
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
        """将ReasoningStep转换为可序列化的字典
        
        参数:
            step: ReasoningStep对象
            
        返回:
            可序列化的字典
        """
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
        """将ProblemDecomposition转换为可序列化的字典
        
        参数:
            decomp: ProblemDecomposition对象
            
        返回:
            可序列化的字典
        """
        return {
            "original_question": decomp.original_question,
            "reasoning_steps": [self.step_to_dict(s) for s in decomp.reasoning_steps],
            "synthesis_step": decomp.synthesis_step,
            "current_step_index": decomp.current_step_index
        }
    
    def to_dict(self, response: SocraticResponse) -> Dict:
        """将SocraticResponse转换为字典以便JSON序列化
        
        参数:
            response: SocraticResponse对象
            
        返回:
            字典表示
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
    """便捷函数：无需实例化类即可生成苏格拉底式推理步骤
    
    参数:
        question: 医学问题
        choices: 答案选项字典
        student_answer: 学生选择的答案
        correct_answer: 正确答案
        explanations: 每个选项的可选解释
        conversation_history: 之前的提示轮次
        source_context: RAG检索到的上下文
        api_key: 可选的API密钥
        
    返回:
        问题分解的字典表示
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
    """检查是否应该触发提示
    
    参数:
        student_answer: 学生选择的答案
        correct_answer: 正确答案
        user_message: 可选的用户消息
        
    返回:
        字典 {"should_trigger": 布尔值, "reason": 字符串}
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
        student_answer="B",  # 错误答案
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

