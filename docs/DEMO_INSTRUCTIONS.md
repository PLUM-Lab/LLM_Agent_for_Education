# Demo Instructions - Decompose & Clarify

## Quick Start

### Method 1: Using the Dropdown (Easiest)

1. Open `medical-quiz.html` in your browser (via http://localhost:8000/medical-quiz.html)
2. In the question bank selector dropdown, select **"Demo Questions (Decompose & Clarify)"**
3. The first question will load automatically
4. Follow the steps below for each demo

### Method 2: Direct File Access

If you want to manually load the file:
- The file is located at: `demo_questions.json`
- It contains 2 questions formatted for the UI

## Demo 1: Decompose (Question 1)

**Question:** Appendicitis case

**Steps:**
1. Select answer **B** (wrong answer)
2. Click "Submit"
3. In the thinking prompt box, paste:
   ```
   I'm not really sure what's going on here. The patient has pain and fever, which sounds like an infection. I think we should give antibiotics first to treat the infection, and maybe do surgery later if needed. I don't understand why surgery would be urgent for this.
   ```
4. Click "Submit My Thinking"
5. **Expected:** System will show 3 sub-questions to guide step-by-step reasoning

## Demo 2: Clarify (Question 2)

**Question:** STEMI case

**Steps:**
1. Select answer **A** (wrong answer)
2. Click "Submit"
3. In the thinking prompt box, paste:
   ```
   This is a STEMI based on the ST elevation in leads II, III, and aVF. The patient needs immediate treatment. I know nitroglycerin helps with chest pain in cardiac events, and I understand that STEMI requires urgent intervention. However, I'm thinking we should start with medical management like nitroglycerin first, then consider catheterization if needed. I recognize the urgency but want to try less invasive options first.
   ```
4. Click "Submit My Thinking"
5. **Expected:** System will provide targeted clarification without revealing the answer

## File Structure

- `demo_questions.json` - Questions formatted for UI (2 questions with demo info)

## Notes

- Each question in `demo_questions.json` includes a `demo_info` field with the `student_thinking` text
- You can copy the `student_thinking` text directly from the JSON file
- The system will automatically call the `/evaluate_student_thinking` API endpoint
- Make sure the RAG server is running on port 5000 for full functionality
