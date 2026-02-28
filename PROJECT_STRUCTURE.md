# Project Structure

## 📁 Directory Organization

```
LLM_Agent_for_Education/
├── 📄 Core Application Files
│   ├── medical-quiz.html          # Main quiz application (frontend)
│   ├── rag_server.py              # RAG backend server
│   ├── proactive_question_generator.py  # Question generation logic
│   ├── start.py                   # Unified startup script
│   ├── start.bat                  # Windows batch startup script
│   ├── README.md                  # Main documentation
│   └── .gitignore                 # Git ignore rules
│
├── ⚙️ Configuration
│   ├── api-key.js                 # OpenAI API key (not in git)
│   └── rag-config.js              # RAG server configuration
│
├── 📊 Data Files
│   ├── data/
│   │   ├── qbanks/                # Question bank JSON files
│   │   │   ├── questions.json
│   │   │   ├── qbank_combined.json
│   │   │   ├── generated_surgery_domain_questions.json
│   │   │   └── ... (other question bank files)
│   │   └── indexes/               # RAG index files
│   │       ├── faiss_index.bin
│   │       └── all_chunks.json
│   └── user_logs/                 # Per-user activity logs
│       └── {username}.json
│
├── 🛠️ Tools & Utilities
│   └── tools/                     # HTML utility pages
│       ├── activity_log.html      # View user activity logs
│       ├── usage_stats.html       # View usage statistics
│       ├── diagnose_coverage.html # Diagnose knowledge coverage
│       ├── login.html             # Login page
│       ├── import_profile_to_browser.html
│       ├── question_evaluator.html
│       └── test_*.html            # Test pages
│
├── 🧪 Tests
│   └── tests/                     # Test files
│       ├── test_*.py
│       └── verify_key.py
│
├── 📝 Documentation
│   └── docs/                      # Additional documentation
│       └── *.md, *.txt, *.docx
│
├── 📋 Logs & Temporary Files
│   ├── logs/                      # Log files
│   │   └── *.log, *.txt
│   └── temp/                      # Temporary files
│       └── test files, temp JSONs
│
├── 🚀 Deployment
│   └── deploy/                    # Standalone deployment package
│       ├── medical-quiz.html
│       ├── questions.json
│       ├── qbank_combined.json
│       ├── generated_surgery_domain_questions.json
│       ├── api-key.example.js
│       └── README.md
│
└── 📚 Other Directories
    ├── scripts/                   # Utility scripts
    ├── images/                    # Image assets
    ├── config/                    # Configuration files
    ├── Clinical Guidelines/       # Source PDFs
    └── Qbanks and Practice Exams/ # Question sources
```

## 🔄 Path Updates

After reorganization, the following paths have been updated:

### Frontend (medical-quiz.html)
- Question banks: `questions.json` → `data/qbanks/questions.json`
- Question banks: `qbank_*.json` → `data/qbanks/qbank_*.json`
- Question banks: `generated_*.json` → `data/qbanks/generated_*.json`

### Backend (rag_server.py)
- Index file: `faiss_index.bin` → `data/indexes/faiss_index.bin`
- Chunks file: `all_chunks.json` → `data/indexes/all_chunks.json`

## 📝 Notes

- **Core files** remain in root for easy access
- **Data files** are organized by type (qbanks vs indexes)
- **Tools** are separated from main application
- **Logs** and **temp files** are isolated for easy cleanup
- **Deploy folder** contains standalone deployment package

## 🧹 Cleanup Recommendations

Files in `temp/` and `logs/` can be safely deleted:
- `temp/` - Temporary test files and old profiles
- `logs/` - Old log files (can be regenerated)
