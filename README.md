# Medical Education Quiz System

An AI-powered medical education platform that automatically generates multiple-choice questions from clinical guidelines and provides interactive ChatGPT-based tutoring with RAG (Retrieval-Augmented Generation) support.

## Features

### 🎯 Question Generation
- Automatically generate questions from 40 medical guideline PDFs
- Each option includes explanations (why correct/incorrect)
- Source tracking (PDF name + page number)
- Uses OpenAI GPT-4o-mini to generate high-quality questions

### 💬 ChatGPT Integration
- Ask follow-up questions on any topic
- Full context awareness (question, answer, explanation)
- Medical tutor role with Socratic teaching method
- Source citations from guidelines

### 🔍 RAG (Retrieval-Augmented Generation)
- Two-stage retrieval: FAISS → ColBERTv2 reranking
- Semantic search across all medical guidelines
- Automatic source citations with page numbers
- Falls back to keyword search when server is unavailable

### 📋 Per-user logs (user_logs/)
- Each user has a separate file: `user_logs/{username}.json`
- Logs login/logout events, button clicks, tutor conversations, knowledge profile, and token/cost usage
- No shared “all users in one file”; everything is per user
- Web UI to view: Activity Log page — select a user to see that user’s events (see [Per-user log](#user-activity-log) below)

## Project Structure

```
LLM_Agent_for_Education/
├── medical-quiz.html      # Main UI (single-page application)
├── activity_log.html      # Per-user log viewer (select user to view their events)
├── usage_stats.html       # Token & cost per user
├── rag_server.py          # RAG backend (FAISS + ColBERTv2)
├── proactive_question_generator.py  # Tutor / Socratic hints
├── start.py               # Unified startup script (recommended)
├── start.bat              # Windows quick launch
├── api-key.js             # Your OpenAI API key (not in Git)
├── user_logs/             # Per-user logs (auto-generated, in .gitignore)
├── *.json                 # Generated questions, qbanks, chunks (root)
├── Clinical Guidelines/   # Medical PDF files
├── Qbanks and Practice Exams/  # Question bank PDFs
├── scripts/               # Utility scripts
│   ├── domain_question_generator.py  # Generate questions by domain
│   ├── generate_questions.py         # Generate from PDFs
│   ├── parse_qbank_openai.py         # Parse Amboss PDFs
│   ├── merge_qbanks.py               # Merge question banks
│   └── amboss/           # Amboss-specific scripts
├── tests/                 # Test scripts (API key, etc.)
├── wsl/                   # WSL & port forwarding scripts
├── config/                # Example configs (api-key.example.js, wslconfig.example)
└── docs/                  # Documentation
```

## Quick Start

### Step 1: Configure API Key

1. Copy `api-key.example.js` to `api-key.js`:
   ```bash
   copy api-key.example.js api-key.js
   ```

2. Edit `api-key.js` and add your OpenAI API key:
   ```javascript
   const OPENAI_API_KEY = 'sk-your-actual-key';
   ```

3. Get API key: https://platform.openai.com/account/api-keys

### Step 2: Install Dependencies

```bash
pip install openai langchain-community pypdf flask flask-cors faiss-cpu ragatouille transformers pymupdf
```

### Step 3: Start Servers and UI

#### Method A: Unified Startup Script (Recommended)

**One command starts everything:**

```bash
python start.py
```
Or double-click `start.bat` (Windows)

**What happens:**
- **Windows**: Starts UI (8000) + RAG in WSL with ColBERTv2 (5000), **automatically sets up port forwarding** (UAC may appear once if not running as admin)
- **Linux/WSL**: Starts UI + RAG with ColBERTv2 directly
- First run: ColBERTv2 loads in ~12 seconds
- Access: http://localhost:8000/medical-quiz.html

**Windows options:**
```bash
# Full ColBERTv2 reranking (WSL + auto port forward, UAC may appear once)
python start.py

# Skip WSL - no ColBERTv2, no admin needed
python start.py --no-wsl-rag
```

**Other startup options:**
```bash
# Start only main UI and RAG server
python start.py --ui --rag

# Start only evaluator interface
python start.py --evaluator

# Custom ports
python start.py --ui --port 8000 --evaluator --evaluator-port 8002

# Restart RAG server (stop existing process and restart)
python start.py --restart-rag
```

**After startup:**
- Main UI: http://localhost:8000/medical-quiz.html
- Activity Log (login/logout + profile): http://localhost:8000/activity_log.html (or http://localhost:5000/activity_log.html if only RAG is running)
- Usage Stats (token/cost per user): http://localhost:8000/usage_stats.html
- RAG Server: http://localhost:5000/health (ColBERTv2 when using WSL)
- Evaluator Interface: http://localhost:8001/question_evaluator.html (if started)

**直接在 WSL 里跑（推荐，无端口转发）：**
```bash
# 在 WSL 终端
cd /mnt/d/LLM_Agent_for_Education
./wsl/start_wsl.sh
```
用 Windows 浏览器打开脚本里显示的地址，如 `http://172.x.x.x:8000/medical-quiz.html`，Tutor 即可用。

**Notes:**
- **Windows + WSL**: Port forwarding (`127.0.0.1:5000` → WSL:5000) is automatic. If UAC appears, click Yes once.
- **Windows + --no-wsl-rag**: RAG runs in current process; ColBERTv2 may not load (FAISS-only).
- **Linux/WSL**: Full ColBERTv2 support.
- First run takes 5-10 minutes to build RAG index (one-time only).
- Press `Ctrl+C` to stop all servers.

#### Method B: Start Separately (For Debugging)

**Windows Environment (FAISS only, no ColBERTv2 reranking):**

```bash
# Terminal 1: Start RAG server (FAISS mode only)
python rag_server.py
# Server starts at: http://localhost:5000

# Terminal 2: Start HTTP server
python -m http.server 8000
# Server starts at: http://localhost:8000

# Open in browser
# http://localhost:8000/medical-quiz.html
```

**WSL/Linux Environment (Full functionality, including ColBERTv2):**

```bash
# Terminal 1: Start RAG server (with ColBERTv2 reranking)
python3 rag_server.py
# Server starts at: http://localhost:5000

# Terminal 2: Start HTTP server
python3 -m http.server 8000
# Server starts at: http://localhost:8000

# Open in browser
# http://localhost:8000/medical-quiz.html
```

#### Service Overview

| Service | Port | Description | Access URL | Required |
|---------|------|-------------|------------|----------|
| Main UI | 8000 | Medical quiz system main interface | http://localhost:8000/medical-quiz.html | ✅ Required |
| RAG Server | 5000 | Semantic search API + activity log API | http://localhost:5000/health | ⚠️ Optional (Recommended) |
| Per-user log viewer | 8000 or 5000 | View a selected user’s login/logout and profile events | http://localhost:8000/activity_log.html or :5000/activity_log.html | ⚠️ Optional |
| Evaluator Interface | 8001 | Question quality evaluation tool | http://localhost:8001/question_evaluator.html | ⚠️ Optional |

**When RAG server is unavailable:**
- UI still works normally
- ChatGPT functionality still works, but falls back to keyword search
- Semantic search functionality unavailable

**Verify server status:**
```bash
# Check RAG server
curl http://localhost:5000/health

# Check HTTP server
curl http://localhost:8000/medical-quiz.html
```

## Usage Guide

### Taking Quizzes

1. Read the question and select an answer (A, B, C, or D)
2. Click "Submit" to view feedback
3. Review explanations for each option
4. Use Previous/Next buttons to navigate

### Asking ChatGPT

1. After submitting an answer, enter your question in the chat box
2. Examples:
   - "Why is B incorrect?"
   - "Explain the pathophysiology"
   - "What are the differential diagnoses?"
3. ChatGPT will return answers with citations

### Get Hints vs Break Down Question

Two tutor-style features in the message area work differently:

| | **Get Hints** | **Break Down Question** |
|--|----------------|--------------------------|
| **Where** | Button next to Feedback | Button next to question (when domain/topic is shown) |
| **Requires** | You must **submit an answer first** | No answer required |
| **Input** | One click (no text); means “I need hints” | Sends a fixed line: “Please explain this question and help me understand.” |
| **Backend** | `POST /generate_hints` | `POST /evaluate_student_thinking` |
| **Backend logic** | Uses **question + your (wrong) answer** only; generates Socratic **reasoning steps + goal** (MedTutor-R1). Does **not** analyze your written reasoning. | Sends your “thinking” (default or typed) and **evaluates understanding level**; then returns **decompose** (sub-questions), **clarify** (explanation), or “understood”. |
| **Display** | One Tutor block: “Work through these steps:” + numbered sub-questions + “Goal:”. Only the latest hint block is shown. | Sub-questions, or a clarification paragraph, or a success message, depending on the backend decision. |
| **Afterwards** | You can click Get Hints again for a new set of hints (previous block is replaced). No multi-turn chat. | **Starts a multi-turn guidance dialogue**: you keep typing in the same input and get more sub-questions or clarification until the flow ends. |

- **Get Hints**: “I’ve answered; give me guiding steps (sub-questions + goal) for this question.”
- **Break Down Question**: “Trigger the ‘explain/help me understand’ flow; evaluate my (default) reasoning and start an interactive guidance session.”

### Session and limits

- **Auto-logout:** If there is no activity (mouse, keyboard, scroll, touch) for **15 minutes**, the user is logged out and the session expired message is shown.
- **Cost limit:** Each user is limited to **$3 per hour** (rolling window). When exceeded, the RAG server returns 429 and the UI shows a cost-limit message until the window resets.

### Per-user log (user_logs/)

Login/logout, button clicks, tutor conversations, knowledge profile, and usage are recorded **per user** in separate files.

- **Where it is stored:** `user_logs/{username}.json` (one file per user). The `user_logs/` directory is in `.gitignore`.
- **When it updates:** On sign in/out (including auto-logout after 15 minutes), button clicks, tutor messages, profile saves, and when usage is read.
- **Contents:** Each file has `button_clicks`, `tutor_conversations`, `knowledge_profile`, `usage`, and `events` (login/logout entries with optional profile).
- **View in browser:** Open the Activity Log page (see URLs above), then select a user to view that user’s events. Full format is described in [Per-user log format](#user-activity-log--detailed-format) below.

### Parsing Question Bank PDFs

Parse questions from Amboss PDF:
```bash
python parse_qbank_openai.py --dir "Qbanks and Practice Exams" --output qbank_amboss_openai.json
```

Parameters:
- `--dir`: PDF file directory
- `--output`: Output JSON file path
- `--model`: OpenAI model (default: gpt-5)
- `--batch-size`: Pages per batch (default: 3)
- `--overlap`: Batch overlap pages (default: 2)

### Merging Question Banks

Merge multiple question bank files:
```bash
python merge_qbanks.py
```

### Regenerating Questions

Generate new questions from PDFs:
```bash
python generate_questions.py
```

Configuration (in `generate_questions.py`):
- `questions_per_doc = 5` - Number of questions per PDF
- `model = "gpt-4o-mini"` - OpenAI model
- `chunk_size = 1024` - Tokens per chunk

## Technical Details

### Question Generation Pipeline

```
PDF files → PyPDFLoader → Split into chunks
    → Random selection (5 per document)
    → GPT-4o-mini generates questions
    → Save to questions.json
```

### RAG Pipeline

```
Student question → OpenAI vector
    → FAISS (top 30 candidates)
    → ColBERTv2 reranking (top 5)
    → Add to ChatGPT context
```

### Data Format (questions.json)

```json
{
  "question": "What is the first-line treatment for...?",
  "options": {
    "A": "Option text",
    "B": "Option text",
    "C": "Option text",
    "D": "Option text"
  },
  "correct_answer": "B",
  "explanations": {
    "A": "Incorrect. Reason...",
    "B": "Correct. Reason...",
    "C": "Incorrect. Reason...",
    "D": "Incorrect. Reason..."
  },
  "source": "Guidelines for AAA repair.pdf",
  "source_page": 12,
  "source_chunk": "Original text from PDF..."
}
```

## Dependencies

### Python Dependencies

```bash
pip install openai langchain-community pypdf flask flask-cors faiss-cpu ragatouille transformers pymupdf
```

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Question Generation | GPT-4o-mini | Generate multiple-choice questions |
| PDF Parsing | GPT-5 / GPT-4o-2024-11-20 | Parse Amboss PDFs |
| Embeddings | text-embedding-3-small | Semantic search |
| Reranking | ColBERTv2 | Improve relevance |

### Estimated Costs

| Task | Approximate Cost |
|------|------------------|
| Generate 200 questions | $0.10 - $0.30 |
| Parse Amboss PDF | $0.50 - $2.00 |
| Build RAG index | $0.05 - $0.10 |
| ChatGPT conversation | $0.001 per message |

## Advanced Configuration

### Installing ColBERTv2 Reranker (Optional)

ColBERTv2 reranker requires compiling C++ extensions. There are three approaches:

#### Option 1: Install Visual Studio Build Tools (Windows)

1. **Download Visual Studio Build Tools**
   - Direct download: https://aka.ms/vs/17/release/vs_buildtools.exe
   - Or visit: https://visualstudio.microsoft.com/downloads/
   - Select "Build Tools for Visual Studio 2022"

2. **Select Components During Installation**
   - Run the installer
   - In the "Workloads" tab, check:
     - ✅ **C++ build tools** workload
   - In the right "Installation details", ensure it includes:
     - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
     - ✅ Windows 10/11 SDK (latest version)
     - ✅ C++ CMake tools for Windows
   - Click "Install" (requires ~3-5 GB space)

3. **After Installation**
   - Close all terminal windows
   - Reopen terminal
   - Run: `python start.py`

4. **Verify Installation**
   ```powershell
   where cl
   ```
   If it shows the compiler path, installation is successful.

#### Option 2: Use WSL (Recommended on Windows)

ColBERTv2's C++ extension uses POSIX thread library (pthread.h), which may not be compatible on Windows. Using WSL avoids this issue.

**On Windows: run `python start.py` from Windows** — it automatically starts RAG in WSL and sets up port forwarding. No need to manually open WSL.

**Step 1: Install WSL**

Run in PowerShell (with administrator privileges):
```powershell
wsl --install
```

Or install a specific version:
```powershell
wsl --install -d Ubuntu-22.04
```

**Important**: After installation, you need to **restart your computer**.

**Step 2: Set Up Environment in WSL**

1. Open WSL (Ubuntu)
2. Update system and install dependencies:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y python3 python3-pip python3-venv build-essential git
   ```

3. Upgrade pip:
   ```bash
   python3 -m pip install --upgrade pip setuptools wheel
   ```

4. Install PyTorch:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. Install project dependencies (install in batches):
   ```bash
   pip3 install faiss-cpu
   pip3 install flask flask-cors
   pip3 install openai
   pip3 install langchain-community
   pip3 install pypdf
   pip3 install pymupdf
   pip3 install ragatouille
   ```

**Or use virtual environment (recommended):**
```bash
# Create virtual environment
cd /mnt/d/LLM_Agent_for_Education
python3 -m venv venv_wsl
source venv_wsl/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu flask flask-cors openai langchain-community pypdf pymupdf ragatouille
```

**Step 3: Run from Windows (recommended)**

```bash
cd d:\LLM_Agent_for_Education
python start.py
```
Start.py will spawn RAG in WSL and set up port forwarding. UAC may appear once (click Yes).

**或者直接在 WSL 里运行（推荐，避免端口转发）：**

在 WSL 终端中执行：
```bash
cd /mnt/d/LLM_Agent_for_Education   # 或你的项目路径
./wsl/start_wsl.sh
# 或：python3 start.py
```
启动后会显示 WSL 的 IP（如 `172.31.177.186`）。在 **Windows 浏览器** 中打开：
```
http://<WSL_IP>:8000/medical-quiz.html
```
例如 `http://172.31.177.186:8000/medical-quiz.html`。UI 和 Tutor 都走同一 WSL IP，无需端口转发，ColBERTv2 正常可用。

**Manual port forwarding (if needed):** If `localhost:5000` is unreachable after WSL restart, run `setup_port_forward.bat` as administrator once.

**Notes:**
- Windows drives in WSL are at `/mnt/d/`, `/mnt/c/`, etc.
- WSL2 uses a separate network; port forwarding connects `127.0.0.1:5000` to WSL.

#### Option 3: Use FAISS-Only Mode (Current State)

If you don't want to install compiler or WSL, the system already works:
- ✓ FAISS retrieval works normally
- ✓ System functionality is complete
- ✗ Only missing ColBERTv2 reranking optimization

**Note**: FAISS retrieval is sufficient for use, reranking is just an optimization.

### Rebuilding RAG Index

If you modify chunk size or other configurations, you need to rebuild the index.

#### Method 1: Rebuild via API (Recommended, No Server Restart)

Run in terminal:
```bash
curl -X POST http://localhost:5000/rebuild
```

**Advantages:**
- No need to stop server
- Background rebuild, can continue using system

**Wait time**: ~5-10 minutes

#### Method 2: Restart Server to Rebuild (Complete Rebuild)

1. **Stop Current Server**
   Press `Ctrl+C` in the terminal running `python start.py`

2. **Delete Old Index Files**
   ```bash
   rm -f faiss_index.bin all_chunks.json
   ```

3. **Restart Server**
   ```bash
   python start.py
   ```

The server will automatically detect that index files don't exist and rebuild the index.

**Rebuild Process:**
- Reload all PDF files (from `Clinical Guidelines` directory)
- Split documents with new chunk size (1024 tokens)
- Generate vectors for each chunk (via OpenAI API)
- Build FAISS index
- Save index files

**Cost**: ~$0.05-0.10 (OpenAI Embeddings API)

**Verify Rebuild Success:**
```bash
curl http://localhost:5000/health
```

Should see:
- `chunks_count`: New chunk count
- `reranker`: ColBERTv2 (if installed)

## Troubleshooting

### "Cannot load questions.json"
- Make sure you're running a local server
- Don't open HTML file directly (file:// protocol)
- Run: `python -m http.server 8000`

### "API key not configured"
- Create `api-key.js` from template
- Or enter API key in browser console:
  ```javascript
  localStorage.setItem('openai_api_key', 'sk-your-key');
  ```

### "RAG Server not available" or Tutor shows "Tutor service temporarily unavailable"
- **Windows + WSL**: Ensure port forwarding is set. Run `wsl/setup_port_forward.bat` as administrator once.
- Or run `python start.py` again — it auto-sets up port forwarding (UAC may appear).
- If WSL IP changed after restart, run `wsl/setup_port_forward.bat` again.
- Fallback: `python start.py --no-wsl-rag` (no ColBERTv2, but works without admin).

### 端口有问题 (Port issues)
- **检查端口与连通性**：运行 `powershell -ExecutionPolicy Bypass -File fix_ports.ps1`，会显示 8000/5000 占用、端口转发和连通性。
- **5000 连不上**：WSL 重启后 IP 可能变化。以管理员身份运行 `setup_port_forward.bat` 刷新转发。
- **8000 被占用**：结束占用 8000 的进程后重新运行 `python start.py`，或用 `python -m http.server 8001` 改端口（需改前端访问地址）。
- **一键修复**：先以管理员运行 `wsl/setup_port_forward.bat`，再执行 `python start.py`。

### "Reranker import failed"
- If you see `[!] Reranker import failed`, this is normal
- System will use FAISS-only mode
- To enable reranking, refer to "Installing ColBERTv2 Reranker" section

### Terminal Display Garbled
- This is a PowerShell encoding issue
- Actual output files (questions.json) are correct
- Use Windows Terminal or VS Code terminal for better display

### pip Installation Failed in WSL

If you encounter `pip` installation errors:

1. **Upgrade pip**:
   ```bash
   python3 -m pip install --upgrade pip
   ```

2. **Clear cache**:
   ```bash
   pip cache purge
   ```

3. **Use virtual environment** (recommended):
   ```bash
   python3 -m venv venv_wsl
   source venv_wsl/bin/activate
   pip install ...
   ```

4. **Install in batches**:
   ```bash
   pip3 install faiss-cpu
   pip3 install flask flask-cors
   pip3 install openai
   pip3 install langchain-community
   pip3 install pypdf
   pip3 install ragatouille
   ```

### Compiling ragatouille is Slow
- This is normal, first compilation may take 5-10 minutes
- Please be patient

---

## UI User Guide (Medical Quiz – How to Use)

This section explains how to use the Medical Quiz app for practicing medical questions and getting help from the AI Tutor.

### Getting Started

**1. Sign In**
- Enter your **username** and **password**, then click **Sign in**
- First time? Click **Create account** to register, then sign in
- Demo accounts: admin / admin or student / student

**2. Understand the Screen**

After signing in, the page has two main areas:

**Left panel (top to bottom):**
- **Header:** "Logged in as: [username]", **Sign out** link, **Questions Answered**, **Knowledge Coverage** (e.g. 0% (0/193 subtopics))
- **By domain (accuracy %):** A list of knowledge domains with your accuracy for each
- **Question Bank:** Dropdown to select a question set
- **Question:** The current question with Domain, Topic, and Subtopic info
- **Choices:** Answer options (A, B, C, D, etc.)
- **Buttons:** Submit, Previous, Next, Jump to, Jump
- **Feedback:** Shows correct/wrong and explanations after you submit

**Right panel – Tutor:** AI Tutor, input box, and **Send** button. You can drag the divider between the panels to resize.

### Practicing Questions

1. **Choose a Question Bank** – Use the dropdown to pick a set (e.g. Surgery Domain Questions).
2. **Read and Answer** – Read the question, click one option (A–D), click **Submit**.
3. **View Feedback** – Feedback shows correct/wrong, explanations, and source references.
4. **Navigate** – **Previous** / **Next**; or use **Jump to** + number + **Jump** (or Enter).

### Using the Tutor

- **Explain your reasoning:** Type in the right panel and click **Send**. The Tutor may **Clarify** or **Decompose** (sub-questions) without giving the answer.
- **Break Down Question (Ask Tutor):** For some banks, an **Break Down Question** button appears next to the question; click it for one-click help.
- **Free chat:** You can type any medical question or comment and **Send**; the Tutor has context and can cite sources.

### Your Progress

- **Questions Answered** – Shown in the header.
- **Knowledge Coverage** – Percentage of subtopics mastered (e.g. 0% (0/193 subtopics)). **By domain (accuracy %)** shows per-domain accuracy. Progress is saved between sessions.

### Usage Limit

If you see “Cost limit exceeded” with a countdown, wait until it finishes; the Tutor will work again when the timer reaches zero.

### Quick Tips

| What you want to do       | How                                  |
|----------------------------|--------------------------------------|
| Switch question banks      | Question Bank menu                   |
| Submit an answer           | **Submit**                           |
| Get quick help (some banks)| **Break Down Question** next to question |
| Explain your reasoning     | Type in right panel, **Send**        |
| Jump to a question         | Enter number, **Jump**               |
| Sign out                   | **Sign out** (top left)              |

---

## Per-user log format (user_logs/{username}.json)

Each user has one file: `user_logs/{username}.json`. There is **no shared file**; all data is per user.

### 文件整体结构（每个用户一个文件）

- **一个用户一个文件**，例如 `user_logs/student.json`。
- 文件内包含：`events`（登录/登出）、`button_clicks`、`tutor_conversations`、`knowledge_profile`、`usage`。

**简单示意：**

```
user_logs/
├── student.json    # 该用户的所有记录
├── admin.json
└── ...
每个文件结构：
├── user              "student"
├── events            [ { "event": "login", "timestamp": "...", "profile": {...} }, { "event": "logout", ... }, ... ]
├── button_clicks     [ { "timestamp": "...", "button": "Submit", "question_id": 1 }, ... ]
├── tutor_conversations  [ { "question_id": 1, "question_text": "...", "options": "...", "messages": [...] }, ... ]
├── knowledge_profile  { ... }   # 最新一次快照
└── usage             { "cost_total", "prompt_tokens", "completion_tokens", "total_tokens", "request_count" }
```

### events 里的一条记录（登录/登出）

- **登出记录**：`event`（"logout"）, `timestamp`（UTC）.
- **登录记录**：`event`（"login"）, `timestamp`, `profile`（可选，含 userName, questionsAnswered, knowledgeMap 等）。

**Timezone:** All `timestamp` values are in **UTC**. Example: `2026-02-21T06:48:24.643Z` = 6:48 AM UTC; for Beijing (UTC+8) that is 2:48 PM the same day.

### When is an event written (in `events`)?

| Event   | When it is recorded |
|---------|----------------------|
| **login**  | When the user signs in, or when the page loads with an existing session. |
| **logout** | When the user clicks "Sign out", or when auto-logged out after **15 minutes** of inactivity. |

### Fields in each `events` entry

| Field        | Type   | Description |
|-------------|--------|-------------|
| **event**   | string | `"login"` or `"logout"`. |
| **timestamp** | string | Time in **UTC** (ISO 8601). |
| **profile** | object | **Only on `login`.** Snapshot of student profile from browser. Omitted on `logout`. |

The **profile** object (on login) includes **userName**, **questionsAnswered**, **knowledgeMap** (cumulative), **createdAt**, **lastUpdated**. **knowledgeMap** keys are `"Domain->Topic->Subtopic"` with **status**, **questionsAttempted**, **questionsCorrect**, **correctQuestionIds**, etc.

The same per-user file also stores **button_clicks**, **tutor_conversations**, **knowledge_profile**, and **usage** (tokens/cost).

---

## Demo Instructions (Decompose & Clarify)

### Quick Start

**Method 1: Using the Dropdown (Easiest)**

1. Open `medical-quiz.html` (e.g. http://localhost:8000/medical-quiz.html)
2. In the question bank selector, select **"Demo Questions (Decompose & Clarify)"**
3. Follow the steps below for each demo

**Method 2: Direct File Access** – The file is `demo_questions.json` (2 questions for the UI).

### Demo 1: Decompose (Question 1 – Appendicitis)

1. Select answer **B** (wrong), click "Submit"
2. In the thinking prompt box, paste the suggested text (e.g. uncertainty about infection vs surgery)
3. Click "Submit My Thinking"
4. **Expected:** System shows sub-questions to guide step-by-step reasoning

### Demo 2: Clarify (Question 2 – STEMI)

1. Select answer **A** (wrong), click "Submit"
2. In the thinking prompt, paste the suggested text (e.g. understanding STEMI but preferring medical management first)
3. Click "Submit My Thinking"
4. **Expected:** System provides targeted clarification without revealing the answer

**Notes:** Each question in `demo_questions.json` can include a `demo_info` field with `student_thinking` text. The system calls `/evaluate_student_thinking`. Ensure the RAG server is running on port 5000.

---

## Security Notes

- `api-key.js` is in `.gitignore` - your key will not be uploaded
- Never share your API key
- Set usage limits in OpenAI dashboard
- Key is stored in browser's localStorage for UI use

## License

MIT License - For educational purposes only

## Author

AI for Education (Wenchao Qin)
