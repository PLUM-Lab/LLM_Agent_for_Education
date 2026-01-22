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

## Project Structure

```
LLM_Agent_for_Education/
├── medical-quiz.html      # Main UI (single-page application)
├── questions.json         # Generated questions (200 questions)
├── generate_questions.py  # Question generation script
├── parse_qbank_openai.py  # Amboss PDF parsing script
├── merge_qbanks.py        # Question bank merging script
├── rag_server.py          # RAG backend (FAISS + ColBERTv2)
├── start.py              # Unified startup script (recommended)
├── start.bat              # Windows quick launch
├── api-key.js             # Your OpenAI API key (not in Git)
├── api-key.example.js     # API key template
├── README.md              # This file
├── Clinical Guidelines/   # 40 medical PDF files
└── Qbanks and Practice Exams/  # Question bank PDF files
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

**Windows:**
```bash
python start.py
```
Or double-click `start.bat`

**Linux/WSL:**
```bash
python3 start.py
```

**Startup Options:**

```bash
# Start all services (default: Main UI + RAG server)
python start.py

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
- RAG Server: http://localhost:5000/health
- Evaluator Interface: http://localhost:8001/question_evaluator.html (if started)

**Notes:**
- Windows environment can start all services, but ColBERTv2 reranker may not be available
- WSL/Linux environment supports full functionality, including ColBERTv2 reranker
- First run takes 5-10 minutes to build RAG index (one-time only)
- Press `Ctrl+C` to stop all servers

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
| RAG Server | 5000 | Semantic search API | http://localhost:5000/health | ⚠️ Optional (Recommended) |
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

#### Option 2: Use WSL (Recommended, More Stable)

ColBERTv2's C++ extension uses POSIX thread library (pthread.h), which may not be compatible on Windows. Using WSL avoids this issue.

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

**Step 3: Run Server in WSL**

```bash
cd /mnt/d/LLM_Agent_for_Education
python3 start.py
```

**Notes:**
- Windows drives in WSL are located at `/mnt/d/`, `/mnt/c/`, etc.
- WSL and Windows share localhost, can access directly
- File paths need `/mnt/d/` prefix to access Windows drives

**Using unified startup script in WSL:**
```bash
cd /mnt/d/LLM_Agent_for_Education
python3 start.py
```

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

### "RAG Server not available"
- If `rag_server.py` is not running, this is normal
- System will fall back to keyword search
- For better results, run RAG server

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

## Security Notes

- `api-key.js` is in `.gitignore` - your key will not be uploaded
- Never share your API key
- Set usage limits in OpenAI dashboard
- Key is stored in browser's localStorage for UI use

## License

MIT License - For educational purposes only

## Author

AI for Education (Wenchao Qin)
