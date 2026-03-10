# Medical Quiz Application - Deployment Package

This folder contains all necessary resources for deploying the Medical Quiz application.

## Files Included

### Core Application
- **medical-quiz.html** - Main application file (single-page web application)

### Configuration
- **../api-key.js** - OpenAI API key configuration (optional, can use localStorage instead)
  - If this file doesn't exist, the application will prompt for API key on first use
  - You can create this file with: `const OPENAI_API_KEY = 'your-api-key-here';`

### Question Bank Data Files
- **questions.json** - Clinical questions generated from medical guidelines
- **qbank_combined.json** - Combined question bank (Surgery + Amboss)
- **generated_surgery_domain_questions.json** - Surgery domain questions (579 questions)

## Deployment Instructions

### Recommended: Single-Port Deployment (UI + RAG API)

From the **project root** (parent of `deploy/`), run:

```bash
cd LLM_Agent_for_Education
python start.py
```

- One server listens on **port 8000** (configurable with `--port`).
- Serves the quiz UI (e.g. `/medical-quiz.html`), static files, and all RAG/Tutor APIs on the same port.
- All RAG features work: search, hints, chat, user logs, cost limit, etc., with no CORS issues.

Then open: `http://localhost:8000/` or `http://localhost:8000/medical-quiz.html`.

For production, point your reverse proxy (e.g. Nginx) or PaaS to this single port.

### Option 2: Static-Only (No Tutor/RAG)

If you only need the quiz UI without the backend:

1. From project root or this folder: `python -m http.server 8000`
2. Open: `http://localhost:8000/medical-quiz.html`

Tutor, RAG search, and user logging will not work without the backend (use `python start.py` instead).

### Option 3: Web Server Deployment

1. Copy the whole project (or at least `medical-quiz.html`, `rag_server.py`, `data/`, and dependencies) to your server.
2. Run `python start.py` (or run `rag_server.py` with port 8000) so one process serves both UI and API.
3. Put a reverse proxy (Nginx/Apache) in front of that single port if needed.

### Option 4: Static Hosting (Limited)

You can host only the HTML/JS/JSON on GitHub Pages, Netlify, etc., but **Tutor and RAG will not work** unless you deploy the backend separately and set CORS/API URL. For full functionality, use single-port deployment above.

## Features

- ??Standalone deployment (all resources in one folder)
- ??No external dependencies (except optional API key)
- ??Works offline (after initial load)
- ??Responsive design (mobile and desktop)
- ??Multiple question banks
- ??Student progress tracking (localStorage)
- ??AI tutor integration (OpenAI API)

## Notes

- **Single port**: One process (port 8000) serves both the quiz UI and all RAG/Tutor APIs; no separate RAG port needed.
- Keep project structure (e.g. `data/qbanks/`, `data/indexes/`) when deploying.
- API key: `api-key.js` in project root or enter in the UI.

## Troubleshooting

### Questions not loading?
- Check browser console for errors
- Ensure JSON files are accessible (check file permissions)
- Verify web server is serving JSON files with correct MIME type (`application/json`)

### API key issues?
- Create `api-key.js` file with: `const OPENAI_API_KEY = 'your-key';`
- Or enter API key manually when prompted in the application

### CORS errors?
- Ensure you're accessing via HTTP server, not `file://` protocol
- Check that your web server allows CORS if accessing from different domain
